"""
Global keyboard hotkeys for Linux using /dev/input/event*.
This does not depend on terminal or window focus.

Notes:
- Linux only.
- Requires read permission for /dev/input/event* (usually root or input group).
- Independent of Wayland/X11 because it reads evdev devices directly.
"""

from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EvdevHotkeyConfig:
    device: str = "auto"  # "/dev/input/eventX" or "auto"
    grab: bool = False    # Enable exclusive grab (use with care).


class EvdevHotkeys:
    """
    Background thread listens for key-down events and triggers callback(ch: str).

    - ch is a lowercase single character, e.g. 'r','k','p','q'.
    """

    def __init__(self, cfg: EvdevHotkeyConfig, callback: Callable[[str], None]) -> None:
        self.cfg = cfg
        self.callback = callback
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._device_path: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        # Resolve/validate device in main thread to fail fast.
        path = self._resolve_device_path()
        self._device_path = path
        try:
            from evdev import InputDevice  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("python-evdev is not installed. Install it with `pip install evdev`.") from e
        try:
            _ = InputDevice(path)
        except PermissionError as e:
            raise PermissionError(
                f"No permission to read {path}. Run with sudo, or add your user to input group and re-login."
            ) from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Device not found: {path}") from e
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass
        self._thread = None

    def _resolve_device_path(self) -> str:
        dev = str(self.cfg.device).strip()
        if dev and dev != "auto":
            return dev
        try:
            from evdev import InputDevice, list_devices  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "python-evdev is not installed. Install it with `pip install evdev`."
            ) from e

        # 1) Prefer /dev/input/by-id/*event-kbd (more stable naming).
        by_id = [p for p in sorted(glob.glob("/dev/input/by-id/*event-kbd")) if os.path.exists(p)]
        if by_id:
            # Avoid selecting foot switches/special inputs by default.
            prefer = [p for p in by_id if "footswitch" not in os.path.basename(p).lower()]
            return (prefer[0] if prefer else by_id[0])

        # 2) Then try evdev's list_devices().
        try:
            devs = list_devices()
        except Exception:
            devs = []

        # 3) If list_devices() is empty, fallback to glob on /dev/input/event*.
        if not devs:
            devs = sorted(glob.glob("/dev/input/event*"))

        if not devs:
            raise FileNotFoundError("No /dev/input/event* devices found (may be unmapped in container).")

        # Auto-pick a device that looks like a keyboard.
        for path in devs:
            try:
                d = InputDevice(path)
                name = (d.name or "").lower()
                if "keyboard" in name or "kbd" in name:
                    return path
            except Exception:
                continue
        # Fallback: first device (may be wrong; explicit --evdev_device is recommended).
        return devs[0]

    def _loop(self) -> None:
        try:
            from evdev import InputDevice, categorize, ecodes  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "python-evdev is not installed. Install it with `pip install evdev`."
            ) from e

        path = self._device_path or self._resolve_device_path()
        dev = InputDevice(path)
        if bool(self.cfg.grab):
            try:
                dev.grab()
            except Exception:
                # Grab failure is non-fatal.
                pass

        # Keycode mapping: KEY_R -> 'r'
        def keycode_to_char(code: str) -> Optional[str]:
            c = str(code)
            if c.startswith("KEY_") and len(c) == 5:  # KEY_A .. KEY_Z
                return c[-1].lower()
            if c.startswith("KEY_") and len(c) == 6:  # KEY_0 .. KEY_9
                return c[-1]
            return None

        for event in dev.read_loop():
            if self._stop.is_set():
                break
            if event.type != ecodes.EV_KEY:
                continue
            try:
                ke = categorize(event)
                # key_down=1
                if int(getattr(ke, "keystate", 0)) != 1:
                    continue
                kc = getattr(ke, "keycode", None)
                # keycode may be a list (e.g. shift+key).
                if isinstance(kc, (list, tuple)):
                    for one in kc:
                        ch = keycode_to_char(str(one))
                        if ch is not None:
                            self.callback(ch)
                            break
                else:
                    ch = keycode_to_char(str(kc))
                    if ch is not None:
                        self.callback(ch)
            except Exception:
                continue


