from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Tuple

import numpy as np

from deploy_real.adapters.body.vdmocap_adapter import (
    VdmocapBodyConfig,
    VdmocapBodyReader,
    VdmocapRuntimeConfig,
    build_vdmocap_runtime,
)
from deploy_real.adapters.body.slimevr_adapter import (
    SlimevrBodyConfig,
    SlimevrBodyReader,
)
from deploy_real.adapters.hand.vdhand_adapter import (
    VdhandConfig,
    VdhandReader,
    VdhandRuntimeConfig,
    build_vdhand_runtime,
    build_vdhand_tracking,
)
from deploy_real.adapters.hand.manus_adapter import (
    ManusConfig,
    ManusReader,
    ManusRuntimeConfig,
    build_manus_runtime,
)
from deploy_real.common.keyboard_toggle import KeyboardToggle
from deploy_real.publishers.redis_twist2_publisher import publish_twist2_step
from deploy_real.publishers.zmq_pose_publisher import (
    close_zmq_pose_publisher,
    create_zmq_pose_publisher,
    publish_zmq_pose_step,
)


# -----------------------------
# Shared infrastructure helpers
# -----------------------------

def ensure_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    deploy_real_root = repo_root / "deploy_real"
    gmr_root = repo_root / "GMR"
    for p in [str(repo_root), str(deploy_real_root), str(gmr_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def normalize_keyboard_backend(cfg: Any, policy: str) -> None:
    allowed_kb_backends = {"none", "stdin", "evdev", "both"}
    if cfg.keyboard_backend not in allowed_kb_backends:
        print(
            f"[{policy}_pipeline] [WARN] Invalid keyboard_backend="
            f"{cfg.keyboard_backend}, fallback to none"
        )
        cfg.keyboard_backend = "none"


def resolve_sources(state: Any, cfg: Any) -> Tuple[str, int, int, str, int, int]:
    dst_ip_body = str(cfg.dst_ip_body).strip() or str(cfg.dst_ip)
    dst_port_body = int(cfg.dst_port_body) if int(cfg.dst_port_body) > 0 else int(cfg.dst_port)
    mocap_index_body = int(cfg.mocap_index_body) if int(cfg.mocap_index_body) >= 0 else int(cfg.mocap_index)
    dst_ip_hand = str(cfg.dst_ip_hand).strip() or str(cfg.dst_ip)
    dst_port_hand = int(cfg.dst_port_hand) if int(cfg.dst_port_hand) > 0 else int(cfg.dst_port)
    mocap_index_hand = int(cfg.mocap_index_hand) if int(cfg.mocap_index_hand) >= 0 else int(cfg.mocap_index)
    state.split_sources = (dst_ip_body != dst_ip_hand) or (dst_port_body != dst_port_hand) or (mocap_index_body != mocap_index_hand)
    state.context["resolved_sources"] = {
        "dst_ip_body": dst_ip_body,
        "dst_port_body": dst_port_body,
        "mocap_index_body": mocap_index_body,
        "dst_ip_hand": dst_ip_hand,
        "dst_port_hand": dst_port_hand,
        "mocap_index_hand": mocap_index_hand,
        "split_sources": bool(state.split_sources),
    }
    return dst_ip_body, dst_port_body, mocap_index_body, dst_ip_hand, dst_port_hand, mocap_index_hand


def init_redis_client(components: Dict[str, Any], cfg: Any) -> None:
    try:
        import redis  # type: ignore

        client = redis.Redis(host=str(cfg.redis_ip), port=6379, db=0, decode_responses=False)
        client.ping()
        components["redis_client"] = client
        components["redis_connected"] = True
    except Exception as e:  # pragma: no cover - env dependent
        components["redis_error"] = str(e)


def init_source_readers(
    components: Dict[str, Any],
    cfg: Any,
    selection: Any,
    *,
    dst_ip_body: str,
    dst_port_body: int,
    mocap_index_body: int,
    dst_ip_hand: str,
    dst_port_hand: int,
    mocap_index_hand: int,
) -> None:
    if selection.body_adapter.endswith("vdmocap_adapter"):
        try:
            br = VdmocapBodyReader(
                VdmocapBodyConfig(
                    dst_ip=dst_ip_body,
                    dst_port=dst_port_body,
                    mocap_index=mocap_index_body,
                    local_port=int(cfg.local_port),
                    world_space=int(cfg.world_space),
                )
            )
            br.initialize()
            components["body_reader"] = br
            components["body_sdk_ready"] = True
        except Exception as e:  # pragma: no cover - env dependent
            components["sdk_error"] = str(e)
    elif selection.body_adapter.endswith("slimevr_adapter"):
        try:
            br = SlimevrBodyReader(
                SlimevrBodyConfig(
                    vmc_ip=str(cfg.vmc_ip),
                    vmc_port=int(cfg.vmc_port),
                    vmc_timeout_s=float(cfg.vmc_timeout_s),
                    vmc_rot_mode=str(cfg.vmc_rot_mode),
                    vmc_use_fk=bool(cfg.vmc_use_fk),
                    vmc_use_viewer_fk=bool(cfg.vmc_use_viewer_fk),
                    vmc_fk_skeleton=str(cfg.vmc_fk_skeleton),
                    vmc_bvh_path=str(cfg.vmc_bvh_path),
                    vmc_bvh_scale=float(cfg.vmc_bvh_scale),
                )
            )
            br.initialize()
            components["body_reader"] = br
            components["body_sdk_ready"] = True
        except Exception as e:  # pragma: no cover - env dependent
            components["sdk_error"] = str(e)
    if selection.hand_adapter.endswith("vdhand_adapter"):
        try:
            hr = VdhandReader(
                VdhandConfig(
                    dst_ip=dst_ip_hand,
                    dst_port=dst_port_hand,
                    mocap_index=mocap_index_hand,
                    local_port=int(cfg.local_port),
                    hands=cfg.hands,
                )
            )
            hr.initialize()
            components["hand_reader"] = hr
            components["hand_sdk_ready"] = True
        except Exception as e:  # pragma: no cover - env dependent
            components["hand_sdk_error"] = str(e)
    elif selection.hand_adapter.endswith("manus_adapter"):
        try:
            hr = ManusReader(
                ManusConfig(
                    address=str(cfg.manus_address),
                    left_glove_sn=(str(cfg.manus_left_sn).strip() or None),
                    right_glove_sn=(str(cfg.manus_right_sn).strip() or None),
                    auto_assign=bool(cfg.manus_auto_assign),
                    recv_timeout_ms=int(cfg.manus_recv_timeout_ms),
                    flip_x=bool(cfg.manus_flip_x),
                    hands=str(cfg.hands),
                )
            )
            hr.initialize()
            components["hand_reader"] = hr
            components["hand_sdk_ready"] = True
        except Exception as e:  # pragma: no cover - env dependent
            components["hand_sdk_error"] = str(e)
    components["sdk_ready"] = bool(components.get("body_sdk_ready", False) and components.get("hand_sdk_ready", False))


def init_keyboard_toggle(components: Dict[str, Any], cfg: Any, KeyboardToggle: Any) -> None:
    kb_backend = str(cfg.keyboard_backend)
    if bool(cfg.keyboard_toggle_send) and kb_backend != "none":
        kb = KeyboardToggle(
            enable=True,
            toggle_send_key=str(cfg.toggle_send_key),
            hold_key=str(cfg.hold_position_key),
            backend=kb_backend,
            evdev_device=str(cfg.evdev_device),
            evdev_grab=bool(cfg.evdev_grab),
        )
        kb.start()
        components["keyboard_toggle"] = kb


def stage_read_body_frame(state: Any) -> None:
    comps = state.context["components"]
    assert isinstance(comps, dict)
    br = comps.get("body_reader", None)
    if br is None:
        state.context["body_frame"] = None
        state.context["body_frame_status"] = "no_reader"
        state.context["stage_read_body_frame_ts"] = time.time()
        return
    out = br.read_frame()
    if bool(out.get("ok", False)):
        state.context["body_frame"] = out.get("body_frame")
        state.context["body_frame_hand"] = out.get("hand_frame")
        state.context["body_frame_status"] = "ok"
        state.frame_index = int(out.get("frame_index", state.frame_index))
    else:
        state.context["body_frame"] = None
        state.context["body_frame_hand"] = None
        state.context["body_frame_status"] = str(out.get("reason", "unknown"))
    state.context["stage_read_body_frame_ts"] = time.time()


def stage_read_hand_frame(state: Any) -> None:
    comps = state.context["components"]
    assert isinstance(comps, dict)
    hr = comps.get("hand_reader", None)
    if hr is None:
        state.context["hand_frame"] = None
        state.context["hand_frame_status"] = "no_reader"
        state.context["stage_read_hand_frame_ts"] = time.time()
        return
    if not state.split_sources:
        body_hands = state.context.get("body_frame_hand", None)
        if isinstance(body_hands, dict):
            state.context["hand_frame"] = body_hands
            state.context["hand_frame_status"] = "ok"
            state.last_hand_frame_time = time.time()
            state.context["stage_read_hand_frame_ts"] = time.time()
            return
    out = hr.read_frame()
    if bool(out.get("ok", False)):
        state.context["hand_frame"] = out.get("hand_frame")
        state.context["hand_frame_status"] = "ok"
        state.hand_frame_index = int(out.get("frame_index", state.hand_frame_index))
        state.last_hand_frame_time = time.time()
    else:
        state.context["hand_frame"] = None
        state.context["hand_frame_status"] = str(out.get("reason", "unknown"))
    state.context["stage_read_hand_frame_ts"] = time.time()


def stage_apply_keyboard_state(state: Any) -> None:
    comps = state.context["components"]
    assert isinstance(comps, dict)
    kb = comps.get("keyboard_toggle", None)
    if kb is not None:
        send_enabled, hold_enabled, exit_requested = kb.get_extended_state()
        state.send_enabled = bool(send_enabled)
        state.hold_enabled = bool(hold_enabled and send_enabled)
        state.exit_requested = bool(exit_requested)
    state.context["stage_apply_keyboard_state_ts"] = time.time()


def stage_step_rate_limiter(state: Any, cfg: Any) -> None:
    if cfg.target_fps > 1e-6:
        time.sleep(max(0.0, 1.0 / cfg.target_fps))
    state.context["stage_step_rate_limiter_ts"] = time.time()


def cleanup_components(state: Any) -> None:
    comps = state.context.get("components", None)
    if not isinstance(comps, dict):
        return
    kb = comps.get("keyboard_toggle", None)
    if kb is not None and hasattr(kb, "stop"):
        try:
            kb.stop()
        except Exception:
            pass
    br = comps.get("body_reader", None)
    if br is not None and hasattr(br, "close"):
        try:
            br.close()
        except Exception:
            pass
    hr = comps.get("hand_reader", None)
    if hr is not None and hasattr(hr, "close"):
        try:
            hr.close()
        except Exception:
            pass


def _build_motion_runtimes(cfg: Any, selection: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
    body_runtime = build_vdmocap_runtime(
        VdmocapRuntimeConfig(
            format=str(cfg.format),
            actual_human_height=float(cfg.actual_human_height),
            smooth=bool(cfg.smooth),
            smooth_window_size=int(cfg.smooth_window_size),
            safe_idle_pose_id=str(cfg.safe_idle_pose_id),
        )
    )
    if selection.hand_adapter.endswith("manus_adapter"):
        hand_runtime = build_manus_runtime(ManusRuntimeConfig())
    else:
        hand_runtime = build_vdhand_runtime(
            VdhandRuntimeConfig(
                hand_fk=bool(cfg.hand_fk),
                hand_fk_end_site_scale=str(cfg.hand_fk_end_site_scale),
                hand_no_csv_transform=True,
                csv_geo_to_bvh_official=True,
                csv_apply_bvh_rotation=True,
                hands=str(cfg.hands),
            )
        )
    return body_runtime, hand_runtime


def _apply_safe_idle_from_runtime(state: Any, body_runtime: Dict[str, Any], *, use_numpy: bool = False) -> None:
    safe_seq = body_runtime.get("safe_idle_body_seq_35", [])
    if isinstance(safe_seq, list) and len(safe_seq) > 0:
        if use_numpy:
            state.safe_idle_body_35 = list(np.asarray(safe_seq[-1], dtype=float).reshape(-1).tolist())
        else:
            state.safe_idle_body_35 = list(safe_seq[-1])
        state.context["safe_idle_body_seq_35"] = safe_seq
    else:
        state.context["safe_idle_body_seq_35"] = [list(state.safe_idle_body_35)]


def _retarget_body_frame(comps: Dict[str, Any], cfg: Any, fr: Any) -> tuple[Any, Any, str]:
    if fr is None or (not bool(comps.get("gmr_ready", False))):
        return None, None, "no_body_or_gmr"
    try:
        fr_local = dict(fr)
        fr_local = comps["apply_bvh_like_coordinate_transform"](fr_local, pos_unit="m", apply_rotation=True)
        fr_gmr = comps["gmr_rename_and_footmod"](fr_local, fmt=str(cfg.format))
        qpos = comps["retargeter"].retarget(fr_gmr, offset_to_ground=bool(cfg.offset_to_ground))
        return qpos, fr_gmr, "ok"
    except Exception as e:
        return None, None, f"error:{e}"


def _compute_hand_activity(state: Any, cfg: Any) -> tuple[int, str, bool]:
    now_ms = int(time.time() * 1000)
    hands_mode = str(cfg.hands).lower()
    hand_fresh = False
    if state.last_hand_frame_time is not None:
        hand_fresh = (time.time() - float(state.last_hand_frame_time)) <= float(cfg.hand_source_timeout_s)
    is_active = bool(state.send_enabled and (not state.hold_enabled) and hand_fresh and hands_mode != "none")
    return now_ms, hands_mode, is_active


def _build_policy_components(*, include_zmq: bool) -> Dict[str, Any]:
    components: Dict[str, Any] = {
        "redis_client": None,
        "redis_connected": False,
        "sdk_ready": False,
        "body_sdk_ready": False,
        "hand_sdk_ready": False,
        "gmr_ready": False,
        "body_reader": None,
        "hand_reader": None,
        "keyboard_toggle": None,
    }
    if include_zmq:
        components.update(
            {
                "zmq_ready": False,
                "zmq_publisher": None,
                "zmq_mailbox_lock": None,
                "zmq_mailbox_latest_packed": None,
                "zmq_mailbox_latest_idx": -1,
                "zmq_sender_stop": None,
                "zmq_sender_thread": None,
                "retargeter": None,
            }
        )
    return components


def _init_common_policy_components(
    *,
    state: Any,
    cfg: Any,
    selection: Any,
    policy: str,
    include_zmq: bool,
    safe_idle_use_numpy: bool,
) -> Dict[str, Any]:
    ensure_import_paths()
    normalize_keyboard_backend(cfg, policy)
    components = _build_policy_components(include_zmq=include_zmq)
    dst_ip_body, dst_port_body, mocap_index_body, dst_ip_hand, dst_port_hand, mocap_index_hand = resolve_sources(state, cfg)
    init_redis_client(components, cfg)
    init_source_readers(
        components,
        cfg,
        selection,
        dst_ip_body=dst_ip_body,
        dst_port_body=dst_port_body,
        mocap_index_body=mocap_index_body,
        dst_ip_hand=dst_ip_hand,
        dst_port_hand=dst_port_hand,
        mocap_index_hand=mocap_index_hand,
    )
    try:
        from deploy_real.pose_csv_loader import apply_bvh_like_coordinate_transform, apply_geo_to_bvh_official, gmr_rename_and_footmod
        from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore
        from general_motion_retargeting import human_head_to_robot_neck  # type: ignore

        components["retargeter"] = GMR(
            src_human=f"bvh_{cfg.format}",
            tgt_robot="unitree_g1",
            actual_human_height=float(cfg.actual_human_height),
        )
        components["apply_geo_to_bvh_official"] = apply_geo_to_bvh_official
        components["apply_bvh_like_coordinate_transform"] = apply_bvh_like_coordinate_transform
        components["gmr_rename_and_footmod"] = gmr_rename_and_footmod
        components["human_head_to_robot_neck"] = human_head_to_robot_neck
        body_runtime, hand_runtime = _build_motion_runtimes(cfg, selection)
        components.update(body_runtime)
        components.update(hand_runtime)
        _apply_safe_idle_from_runtime(state, body_runtime, use_numpy=safe_idle_use_numpy)
        init_keyboard_toggle(components, cfg, KeyboardToggle)
        components["gmr_ready"] = True
    except Exception as e:  # pragma: no cover - env dependent
        components["gmr_error"] = str(e)
        components["gmr_ready"] = False
    state.context["components"] = components
    state.context["stage_init_components_ts"] = time.time()
    return components


def _resolve_pack_pose_message() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    gr00t_root = repo_root.parent / "GR00T-WholeBodyControl"
    if gr00t_root.exists() and str(gr00t_root) not in sys.path:
        sys.path.insert(0, str(gr00t_root))
    pack_pose_message = None
    try:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # type: ignore
    except Exception:
        try:
            from gear_sonic.utils.zmq_utils import pack_pose_message  # type: ignore
        except Exception:
            pack_pose_message = None
    return pack_pose_message


def _enable_sonic_zmq(components: Dict[str, Any], cfg: Any) -> None:
    pack_pose_message = _resolve_pack_pose_message()
    zmq_step = int(cfg.zmq_frame_index_step) if int(cfg.zmq_frame_index_step) > 0 else (
        max(1, int(round(100.0 / float(cfg.target_fps)))) if float(cfg.target_fps) > 1e-6 else 1
    )
    if pack_pose_message is not None:
        components["zmq_publisher"] = create_zmq_pose_publisher(
            bind_host=str(cfg.zmq_bind_host),
            port=int(cfg.zmq_pose_port),
            topic=str(cfg.zmq_pose_topic),
            pack_pose_message=pack_pose_message,
        )
        components["zmq_frame_index_step"] = int(zmq_step)
        components["zmq_use_mailbox"] = bool(cfg.zmq_use_mailbox)
        if bool(cfg.zmq_use_mailbox):
            lock = threading.Lock()
            stop_ev = threading.Event()
            components["zmq_mailbox_lock"] = lock
            components["zmq_sender_stop"] = stop_ev

            def _sender_loop() -> None:
                while not stop_ev.is_set():
                    packed = None
                    with lock:
                        latest = components.get("zmq_mailbox_latest_packed", None)
                        if latest is not None:
                            packed = latest
                            components["zmq_mailbox_latest_packed"] = None
                    if packed is not None:
                        try:
                            pub = components.get("zmq_publisher", None)
                            if isinstance(pub, dict):
                                sock = pub.get("socket", None)
                                if sock is not None:
                                    sock.send(packed)
                        except Exception:
                            pass
                    else:
                        time.sleep(0.001)

            th = threading.Thread(target=_sender_loop, daemon=True, name="sonic-zmq-mailbox-sender")
            th.start()
            components["zmq_sender_thread"] = th
        components["zmq_ready"] = True
    else:
        components["zmq_error"] = "pack_pose_message_not_found"


def _compute_retarget_outputs(
    *,
    state: Any,
    cfg: Any,
    comps: Dict[str, Any],
    qpos: Any,
    fr_gmr: Any,
    fallback_with_joints_when_short: bool,
    apply_smoothing: bool,
    include_quat_and_joint29: bool,
) -> Dict[str, Any]:
    # Keep last published pose on no-update frames. Legacy scripts skip such frames
    # instead of forcing safe-idle, otherwise legs can repeatedly dip into squat.
    body_35 = list(state.last_pub_body_35)
    neck_2 = list(state.last_pub_neck_2)
    joint_pos29 = np.asarray(
        state.context.get("retarget_joint_pos29_raw", np.zeros((29,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(29)
    body_quat_w = np.asarray(
        state.context.get("retarget_body_quat_w", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        dtype=np.float32,
    ).reshape(4)
    if state.hold_enabled:
        body_35 = list(state.last_pub_body_35)
        neck_2 = list(state.last_pub_neck_2)
    elif state.send_enabled and (qpos is not None):
        dt = 1.0 / 60.0 if state.last_time is None else max(1e-4, float(time.time() - state.last_time))
        state.last_time = time.time()
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if include_quat_and_joint29 and q.shape[0] >= 36:
            joint_pos29 = np.asarray(q[7:36], dtype=np.float32).reshape(29)
            body_quat_w = _safe_unit_quat_wxyz(np.asarray(q[3:7], dtype=np.float32))
        if state.last_qpos is None:
            body_35 = list(state.safe_idle_body_35)
        elif q.shape[0] >= 36 and np.asarray(state.last_qpos).reshape(-1).shape[0] >= 36:
            lq = np.asarray(state.last_qpos, dtype=np.float32).reshape(-1)
            body_35 = np.asarray(comps["extract_mimic_obs_whole_body"](q, lq, dt=dt), dtype=np.float32).reshape(-1).tolist()
        elif fallback_with_joints_when_short:
            joints = np.zeros((29,), dtype=np.float32)
            if q.shape[0] >= 36:
                joints = q[7:36].astype(np.float32)
            body_35 = np.concatenate([np.zeros((6,), dtype=np.float32), joints], axis=0).tolist()
        if apply_smoothing:
            sf = comps.get("smooth_filter", None)
            if sf is not None and hasattr(sf, "apply"):
                sm = sf.apply(np.asarray(body_35, dtype=float))
                if sm is not None:
                    body_35 = np.asarray(sm, dtype=float).reshape(-1).tolist()
        state.last_qpos = q.copy()
    return {
        "body_35": body_35,
        "neck_2": neck_2,
        "joint_pos29": joint_pos29,
        "body_quat_w": body_quat_w,
    }


def _store_retarget_outputs(state: Any, out: Dict[str, Any], *, with_sonic_fields: bool) -> None:
    state.context["retarget_body_35"] = out["body_35"]
    state.context["retarget_neck_2"] = out["neck_2"]
    if with_sonic_fields:
        state.context["retarget_joint_pos29_raw"] = out["joint_pos29"]
        state.context["retarget_joint_pos29"] = _mujoco29_to_isaaclab29(out["joint_pos29"])
        state.context["retarget_body_quat_w"] = out["body_quat_w"]


def _resolve_hand_modes(
    *,
    state: Any,
    cfg: Any,
    now_ms: int,
    ht_l: Dict[str, Any],
    ht_r: Dict[str, Any],
) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    mode = "hold" if state.hold_enabled else ("follow" if state.send_enabled else "default")
    hands_mode = str(cfg.hands).lower()
    if hands_mode == "none":
        mode_l = mode_r = "default"
        return mode_l, mode_r, {"is_active": False, "timestamp": now_ms}, {"is_active": False, "timestamp": now_ms}
    return mode, mode, ht_l, ht_r


def _build_hand_tracking_outputs(
    *,
    state: Any,
    cfg: Any,
    comps: Dict[str, Any],
    requires_tracking_symbol: bool,
) -> Dict[str, Any]:
    tracking_fn = comps.get("build_hand_tracking", None)
    if not callable(tracking_fn):
        tracking_fn = build_vdhand_tracking
    if requires_tracking_symbol and (not callable(tracking_fn)):
        now_ms = int(time.time() * 1000)
        return {
            "hand_tracking_left": {"is_active": False, "timestamp": now_ms},
            "hand_tracking_right": {"is_active": False, "timestamp": now_ms},
        }
    now_ms, hands_mode, is_active = _compute_hand_activity(state, cfg)
    ht_l, ht_r = tracking_fn(
        hands_mode=hands_mode,
        is_active=bool(is_active),
        now_ms=now_ms,
        fr_hand=state.context.get("hand_frame", None),
        cfg_hand_fk=bool(cfg.hand_fk),
        runtime=comps,
    )
    return {
        "hand_tracking_left": ht_l,
        "hand_tracking_right": ht_r,
    }


def _store_hand_outputs(state: Any, out: Dict[str, Any]) -> None:
    state.context["hand_tracking_left"] = out["hand_tracking_left"]
    state.context["hand_tracking_right"] = out["hand_tracking_right"]


def _apply_hand_side_mask(
    *,
    hands_mode: str,
    now_ms: int,
    ht_l: Dict[str, Any],
    ht_r: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    out_ht_l = dict(ht_l)
    out_ht_r = dict(ht_r)
    if hands_mode in ["none", "right"]:
        out_ht_l = {"is_active": False, "timestamp": now_ms}
    if hands_mode in ["none", "left"]:
        out_ht_r = {"is_active": False, "timestamp": now_ms}
    return out_ht_l, out_ht_r


def _prepare_hand_publish_payload(
    *,
    state: Any,
    cfg: Any,
    comps: Dict[str, Any],
    now_ms: int,
    ht_l: Dict[str, Any],
    ht_r: Dict[str, Any],
    apply_side_mask: bool,
) -> Dict[str, Any]:
    mode_l, mode_r, ht_l2, ht_r2 = _resolve_hand_modes(state=state, cfg=cfg, now_ms=now_ms, ht_l=dict(ht_l), ht_r=dict(ht_r))
    hands_mode = str(cfg.hands).lower()
    if apply_side_mask:
        ht_l2, ht_r2 = _apply_hand_side_mask(
            hands_mode=hands_mode,
            now_ms=now_ms,
            ht_l=ht_l2,
            ht_r=ht_r2,
        )
    return {
        "mode_l": mode_l,
        "mode_r": mode_r,
        "ht_l": ht_l2,
        "ht_r": ht_r2,
    }


def _write_wuji_hand_redis(
    *,
    client: Any,
    now_ms: int,
    mode_l: str,
    mode_r: str,
    ht_l: Dict[str, Any],
    ht_r: Dict[str, Any],
) -> None:
    robot_key = "unitree_g1_with_hands"
    key_t_action = "t_action"
    key_ht_l = f"hand_tracking_left_{robot_key}"
    key_ht_r = f"hand_tracking_right_{robot_key}"
    key_wuji_mode_l = f"wuji_hand_mode_left_{robot_key}"
    key_wuji_mode_r = f"wuji_hand_mode_right_{robot_key}"
    pipe = client.pipeline()
    pipe.set(key_t_action, now_ms)
    pipe.set(key_wuji_mode_l, str(mode_l))
    pipe.set(key_wuji_mode_r, str(mode_r))
    pipe.set(key_ht_l, json.dumps(ht_l))
    pipe.set(key_ht_r, json.dumps(ht_r))
    pipe.execute()


def _update_last_published_body(state: Any, body_35: Any, neck_2: Any) -> None:
    state.last_pub_body_35 = list(body_35)
    state.last_pub_neck_2 = list(neck_2)


def _publish_twist2_from_payload(
    *,
    client: Any,
    cfg: Any,
    body_35: Any,
    hand_payload: Dict[str, Any],
    now_ms: int,
) -> None:
    publish_twist2_step(
        redis_client=client,
        body_35=list(body_35),
        hand_tracking_left=dict(hand_payload["ht_l"]),
        hand_tracking_right=dict(hand_payload["ht_r"]),
        wuji_mode_left=str(hand_payload["mode_l"]),
        wuji_mode_right=str(hand_payload["mode_r"]),
        now_ms=int(now_ms),
    )


def _build_sonic_zmq_payload(state: Any, cfg: Any, *, step: int, n_frames: int, frame0: int) -> Dict[str, Any]:
    jp = np.asarray(
        state.context.get("retarget_joint_pos29", np.zeros((29,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(1, 29)
    bq = np.asarray(
        state.context.get("retarget_body_quat_w", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        dtype=np.float32,
    ).reshape(1, 4)
    payload: Dict[str, Any] = {
        "joint_pos": np.repeat(jp, n_frames, axis=0),
        "joint_vel": np.zeros((n_frames, 29), dtype=np.float32),
        "body_quat_w": np.repeat(bq, n_frames, axis=0),
        "frame_index": np.asarray([frame0 + i * step for i in range(n_frames)], dtype=np.int64),
        "timestamp_realtime": np.asarray([time.time()] * n_frames, dtype=np.float64),
        "heading_increment": np.asarray([0.0] * n_frames, dtype=np.float32),
        "catch_up": np.asarray([bool(cfg.zmq_catch_up)] * n_frames, dtype=bool),
    }
    return payload


def _send_sonic_zmq_payload(comps: Dict[str, Any], pub: Dict[str, Any], payload: Dict[str, Any], *, frame_index: int) -> str:
    use_mailbox = bool(comps.get("zmq_use_mailbox", False))
    if use_mailbox:
        pack_pose_message = pub.get("pack_pose_message", None)
        topic = str(pub.get("topic", "pose"))
        if pack_pose_message is None:
            return "pack_missing"
        packed = pack_pose_message(payload, topic=topic, version=1)  # type: ignore[misc]
        lock = comps.get("zmq_mailbox_lock", None)
        if lock is not None:
            with lock:
                comps["zmq_mailbox_latest_packed"] = packed
                comps["zmq_mailbox_latest_idx"] = int(frame_index)
        return "ok"
    publish_zmq_pose_step(publisher=pub, payload=payload)
    return "ok"


def _stage_retarget_common(
    *,
    state: Any,
    cfg: Any,
    fallback_with_joints_when_short: bool,
    apply_smoothing: bool,
    with_sonic_fields: bool,
) -> None:
    comps = state.context["components"]
    qpos, fr_gmr, retarget_status = _retarget_body_frame(comps, cfg, state.context.get("body_frame", None))
    state.context["retarget_status"] = retarget_status
    out = _compute_retarget_outputs(
        state=state,
        cfg=cfg,
        comps=comps,
        qpos=qpos,
        fr_gmr=fr_gmr,
        fallback_with_joints_when_short=fallback_with_joints_when_short,
        apply_smoothing=apply_smoothing,
        include_quat_and_joint29=with_sonic_fields,
    )
    _store_retarget_outputs(state, out, with_sonic_fields=with_sonic_fields)
    state.context["stage_retarget_body_ts"] = time.time()


def _stage_build_hand_common(
    *,
    state: Any,
    cfg: Any,
    requires_tracking_symbol: bool,
) -> None:
    comps = state.context["components"]
    out = _build_hand_tracking_outputs(
        state=state,
        cfg=cfg,
        comps=comps,
        requires_tracking_symbol=requires_tracking_symbol,
    )
    _store_hand_outputs(state, out)
    state.context["stage_build_hand_tracking_ts"] = time.time()


def _stage_publish_hand_common(*, state: Any, cfg: Any, mode: str) -> str:
    comps = state.context["components"]
    client = comps.get("redis_client", None)
    if client is None:
        return "redis_unavailable"
    now_ms = int(time.time() * 1000)
    ht_l = state.context.get("hand_tracking_left", {"is_active": False, "timestamp": now_ms})
    ht_r = state.context.get("hand_tracking_right", {"is_active": False, "timestamp": now_ms})
    apply_side_mask = mode == "sonic"
    try:
        hand_payload = _prepare_hand_publish_payload(
            state=state,
            cfg=cfg,
            comps=comps,
            now_ms=now_ms,
            ht_l=ht_l,
            ht_r=ht_r,
            apply_side_mask=apply_side_mask,
        )
        if mode == "twist2":
            body_35 = state.context["retarget_body_35"]
            neck_2 = state.context["retarget_neck_2"]
            _publish_twist2_from_payload(
                client=client,
                cfg=cfg,
                body_35=body_35,
                hand_payload=hand_payload,
                now_ms=now_ms,
            )
            _update_last_published_body(state, body_35, neck_2)
            return "ok"
        _write_wuji_hand_redis(
            client=client,
            now_ms=now_ms,
            mode_l=str(hand_payload["mode_l"]),
            mode_r=str(hand_payload["mode_r"]),
            ht_l=dict(hand_payload["ht_l"]),
            ht_r=dict(hand_payload["ht_r"]),
        )
        return "ok"
    except Exception as e:
        return (f"redis_error:{e}" if mode == "twist2" else f"error:{e}")


def _stage_cleanup_common(*, state: Any, include_zmq: bool) -> None:
    comps = state.context.get("components", None)
    cleanup_components(state)
    if (not include_zmq) or (not isinstance(comps, dict)):
        return
    stop_ev = comps.get("zmq_sender_stop", None)
    if stop_ev is not None and hasattr(stop_ev, "set"):
        try:
            stop_ev.set()
        except Exception:
            pass
    th = comps.get("zmq_sender_thread", None)
    if th is not None and hasattr(th, "join"):
        try:
            th.join(timeout=0.3)
        except Exception:
            pass
    close_zmq_pose_publisher(comps.get("zmq_publisher", None))


# --------------------
# Twist2 stage wrappers
# --------------------

def twist2_stage_init_components(state: Any, cfg: Any, selection: Any) -> None:
    _init_common_policy_components(
        state=state,
        cfg=cfg,
        selection=selection,
        policy="twist2",
        include_zmq=False,
        safe_idle_use_numpy=False,
    )


def twist2_stage_retarget_body(state: Any, cfg: Any) -> None:
    _stage_retarget_common(
        state=state,
        cfg=cfg,
        fallback_with_joints_when_short=True,
        apply_smoothing=True,
        with_sonic_fields=False,
    )


def twist2_stage_build_hand_tracking(state: Any, cfg: Any) -> None:
    _stage_build_hand_common(
        state=state,
        cfg=cfg,
        requires_tracking_symbol=True,
    )


def twist2_stage_publish_redis(state: Any, cfg: Any) -> None:
    stage_status = _stage_publish_hand_common(state=state, cfg=cfg, mode="twist2")
    state.context["publish_status"] = stage_status
    state.context["stage_publish_redis_ts"] = time.time()


def twist2_cleanup_components(state: Any) -> None:
    _stage_cleanup_common(state=state, include_zmq=False)


def _safe_unit_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if (not np.isfinite(n)) or n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


def _mujoco29_to_isaaclab29(joint_pos29: np.ndarray) -> np.ndarray:
    jp = np.asarray(joint_pos29, dtype=np.float32).reshape(29)
    return jp[_MUJOCO_TO_ISAACLAB_DOF].astype(np.float32).copy()


def _mimic35_to_joint_pos29(mimic35: np.ndarray) -> np.ndarray:
    arr = np.asarray(mimic35, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 35:
        out = np.zeros((29,), dtype=np.float32)
        out[: min(29, arr.shape[0])] = arr[: min(29, arr.shape[0])]
        return out
    return arr[6:35].astype(np.float32).copy()


# -------------------
# Sonic stage wrappers
# -------------------

def sonic_stage_init_components(state: Any, cfg: Any, selection: Any) -> None:
    components = _init_common_policy_components(
        state=state,
        cfg=cfg,
        selection=selection,
        policy="sonic",
        include_zmq=True,
        safe_idle_use_numpy=True,
    )
    if bool(components.get("gmr_ready", False)):
        _enable_sonic_zmq(components, cfg)


def sonic_stage_retarget_body(state: Any, cfg: Any) -> None:
    _stage_retarget_common(
        state=state,
        cfg=cfg,
        fallback_with_joints_when_short=False,
        apply_smoothing=False,
        with_sonic_fields=True,
    )


def sonic_stage_build_hand_tracking(state: Any, cfg: Any) -> None:
    _stage_build_hand_common(
        state=state,
        cfg=cfg,
        requires_tracking_symbol=False,
    )


def sonic_stage_publish_body_zmq(state: Any, cfg: Any) -> None:
    comps = state.context["components"]
    pub = comps.get("zmq_publisher", None)
    if not isinstance(pub, dict):
        state.context["zmq_publish_status"] = "unavailable"
        return
    try:
        body_35_pub = np.asarray(state.context.get("retarget_body_35", state.last_pub_body_35), dtype=np.float32).reshape(-1)
        if body_35_pub.shape[0] >= 35:
            state.context["retarget_joint_pos29"] = _mujoco29_to_isaaclab29(_mimic35_to_joint_pos29(body_35_pub))
        n_frames = max(1, int(getattr(cfg, "zmq_num_frames_to_send", 1)))
        step = max(1, int(comps.get("zmq_frame_index_step", 1)))
        frame0 = int(state.zmq_frame_index)
        payload = _build_sonic_zmq_payload(state, cfg, step=step, n_frames=n_frames, frame0=frame0)
        publish_status = _send_sonic_zmq_payload(comps, pub, payload, frame_index=int(state.zmq_frame_index))
        if publish_status != "ok":
            state.context["zmq_publish_status"] = publish_status
            return
        state.zmq_frame_index += max(1, step) * n_frames
        state.context["zmq_publish_status"] = "ok"
    except Exception as e:
        state.context["zmq_publish_status"] = f"error:{e}"


def sonic_stage_publish_hand_redis(state: Any, cfg: Any) -> None:
    state.context["redis_hand_status"] = _stage_publish_hand_common(state=state, cfg=cfg, mode="sonic")


def sonic_cleanup_components(state: Any) -> None:
    _stage_cleanup_common(state=state, include_zmq=True)
