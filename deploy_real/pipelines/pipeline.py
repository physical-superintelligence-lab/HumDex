from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

from deploy_real.common.twist2_state_machine import (
    Twist2LoopState,
    apply_hold_default_and_ramp,
    init_loop_state,
    on_keyboard_state,
)
from deploy_real.pipelines.utils.logging import (
    log_exit_ramp_finished,
    log_exit_requested,
    log_periodic,
    log_pipeline_start,
    log_step,
)
from deploy_real.pipelines.utils.stages import (
    stage_apply_keyboard_state,
    stage_read_body_frame,
    stage_read_hand_frame,
    stage_step_rate_limiter,
)


@dataclass(frozen=True)
class ComponentSelection:
    body_adapter: str
    hand_adapter: str
    policy_core: str
    publisher: str
    keyboard: str = "keyboard_toggle_controller"
    rate_limiter: str = "rate_limiter"
    zmq_publisher: str | None = None


@dataclass
class RuntimeState:
    send_enabled: bool = True
    hold_enabled: bool = False
    exit_requested: bool = False
    frame_index: int = -1
    hand_frame_index: int = -1
    zmq_frame_index: int = 0
    loop_count: int = 0
    last_time: float | None = None
    last_qpos: Any = None
    safe_idle_body_35: List[float] = field(default_factory=lambda: [0.0] * 35)
    last_pub_body_35: List[float] = field(default_factory=lambda: [0.0] * 35)
    last_pub_neck_2: List[float] = field(default_factory=lambda: [0.0, 0.0])
    last_hand_frame_time: float | None = None
    split_sources: bool = False
    context: Dict[str, object] = field(default_factory=dict)


LoadConfigFn = Callable[[List[str], Path], Any]
InitFn = Callable[[RuntimeState, Any, ComponentSelection], None]
StageFn = Callable[[RuntimeState, Any], None]
CleanupFn = Callable[[RuntimeState], None]


class OrchestratedPipeline:
    def __init__(
        self,
        *,
        policy: str,
        selection: ComponentSelection,
        passthrough: List[str],
        stages: Sequence[str],
        load_config_fn: LoadConfigFn,
        init_components_fn: InitFn,
        retarget_body_fn: StageFn,
        build_hand_tracking_fn: StageFn,
        publish_fns: Sequence[StageFn],
        cleanup_fn: CleanupFn,
        config_fields: Sequence[str],
        component_fields: Sequence[str],
        status_fields: Sequence[str],
    ) -> None:
        self.policy = policy
        self.selection = selection
        self.passthrough = passthrough
        self.stages = list(stages)
        self.load_config_fn = load_config_fn
        self.init_components_fn = init_components_fn
        self.retarget_body_fn = retarget_body_fn
        self.build_hand_tracking_fn = build_hand_tracking_fn
        self.publish_fns = list(publish_fns)
        self.cleanup_fn = cleanup_fn
        self.config_fields = list(config_fields)
        self.component_fields = list(component_fields)
        self.status_fields = list(status_fields)
        self.state = RuntimeState()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _ease(alpha: float, ease: str = "cosine") -> float:
        a = float(np.clip(alpha, 0.0, 1.0))
        if str(ease).lower() == "linear":
            return a
        return 0.5 - 0.5 * float(np.cos(np.pi * a))

    def describe(self) -> Dict[str, object]:
        components: Dict[str, object] = {
            "body_adapter": self.selection.body_adapter,
            "hand_adapter": self.selection.hand_adapter,
            "policy_core": self.selection.policy_core,
            "publisher": self.selection.publisher,
            "keyboard": self.selection.keyboard,
            "rate_limiter": self.selection.rate_limiter,
        }
        if self.selection.zmq_publisher is not None:
            components["zmq_publisher"] = self.selection.zmq_publisher
        return {
            "policy": self.policy,
            "stages": list(self.stages),
            "components": components,
            "passthrough": list(self.passthrough),
        }

    def run(self) -> int:
        try:
            cfg = self.load_config_fn(
                passthrough=self.passthrough,
                repo_root=Path(__file__).resolve().parents[2],
            )
            self.state.context["config"] = cfg
            self.state.context["stage_load_config_ts"] = time.time()
            self.init_components_fn(self.state, cfg, self.selection)
            comps = self.state.context["components"]
            assert isinstance(comps, dict)

            log_pipeline_start(
                policy=self.policy,
                cfg=cfg,
                state=self.state,
                comps=comps,
                config_fields=self.config_fields,
                component_fields=self.component_fields,
            )
            safe_idle_seq = self.state.context.get("safe_idle_body_seq_35", [list(self.state.safe_idle_body_35)])
            assert isinstance(safe_idle_seq, list)
            loop_state: Twist2LoopState = init_loop_state(
                last_pub_body_35=list(self.state.last_pub_body_35),
                last_pub_neck_2=list(self.state.last_pub_neck_2),
                safe_idle_body_35=list(self.state.safe_idle_body_35),
                safe_idle_seq_35=safe_idle_seq,
                start_ramp_seconds=float(getattr(cfg, "start_ramp_seconds", 0.0)),
                now_s=time.time(),
            )
            for _ in range(int(getattr(cfg, "max_steps", 1))):
                stage_read_body_frame(self.state)
                stage_read_hand_frame(self.state)
                stage_apply_keyboard_state(self.state)
                send_enabled, hold_enabled, should_break_now = on_keyboard_state(
                    loop=loop_state,
                    send_enabled=bool(self.state.send_enabled),
                    hold_enabled=bool(self.state.hold_enabled),
                    exit_requested=bool(self.state.exit_requested),
                    last_pub_body_35=list(self.state.last_pub_body_35),
                    last_pub_neck_2=list(self.state.last_pub_neck_2),
                    toggle_ramp_seconds=float(getattr(cfg, "toggle_ramp_seconds", 0.0)),
                    exit_ramp_seconds=float(getattr(cfg, "exit_ramp_seconds", 0.0)),
                    now_s=time.time(),
                )
                self.state.send_enabled = bool(send_enabled)
                self.state.hold_enabled = bool(hold_enabled)
                if should_break_now:
                    log_exit_requested(policy=self.policy)
                    break
                self.retarget_body_fn(self.state, cfg)
                out_body, out_neck, exit_ramp_done = apply_hold_default_and_ramp(
                    loop=loop_state,
                    send_enabled=bool(self.state.send_enabled),
                    hold_enabled=bool(self.state.hold_enabled),
                    retarget_body_35=list(self.state.context["retarget_body_35"]),
                    retarget_neck_2=list(self.state.context["retarget_neck_2"]),
                    ramp_ease=str(getattr(cfg, "ramp_ease", "cosine")),
                    now_s=time.time(),
                    ease_fn=self._ease,
                )
                self.state.context["retarget_body_35"] = out_body
                self.state.context["retarget_neck_2"] = out_neck
                self.state.last_pub_body_35 = list(out_body)
                self.state.last_pub_neck_2 = list(out_neck)
                self.build_hand_tracking_fn(self.state, cfg)
                for fn in self.publish_fns:
                    fn(self.state, cfg)
                if exit_ramp_done:
                    log_exit_ramp_finished(policy=self.policy)
                    break
                stage_step_rate_limiter(self.state, cfg)
                self.state.loop_count += 1
                self.state.frame_index += 1
                log_step(policy=self.policy, state=self.state, status_fields=self.status_fields)
                if int(getattr(cfg, "print_every", 0)) > 0 and (self.state.loop_count % int(getattr(cfg, "print_every", 0)) == 0):
                    log_periodic(
                        frame_index=self.state.frame_index,
                        step=self.state.loop_count,
                        now_ms=self._now_ms(),
                    )
            return 0
        finally:
            self.cleanup_fn(self.state)


def resolve_pipeline_spec(policy: str, body_source: str, hand_source: str, passthrough: list[str]) -> Dict[str, object]:
    from deploy_real.pipelines import sonic_policy, twist2_policy

    if policy == "twist2":
        return twist2_policy.describe_pipeline(body_source=body_source, hand_source=hand_source, passthrough=passthrough)
    if policy == "sonic":
        return sonic_policy.describe_pipeline(body_source=body_source, hand_source=hand_source, passthrough=passthrough)
    raise ValueError(f"Unsupported policy: {policy}")


def build_pipeline_instance(policy: str, body_source: str, hand_source: str, passthrough: list[str]) -> Any:
    from deploy_real.pipelines import sonic_policy, twist2_policy

    if policy == "twist2":
        return twist2_policy.build_pipeline(body_source=body_source, hand_source=hand_source, passthrough=passthrough)
    if policy == "sonic":
        return sonic_policy.build_pipeline(body_source=body_source, hand_source=hand_source, passthrough=passthrough)
    raise ValueError(f"Unsupported policy: {policy}")
