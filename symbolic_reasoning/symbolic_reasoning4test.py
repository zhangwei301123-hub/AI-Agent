"""类似 maddpg4test.py 的符号推理运行入口，但不加载神经网络。"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .agent import Decision, ReasoningFacts, SymbolicReasoningAgent
from .entity import EncodedEntity, EncodedSituation, EntityEncoder, TargetDomain, load_situation


@dataclass(frozen=True)
class SymbolicStepResult:
    situation: EncodedSituation
    facts: Dict[str, ReasoningFacts]
    decisions: Dict[str, Decision]
    actions_dict: Dict[str, List[List[Any]]]
    execute_results: Any = None
    rewards: Any = None


class SymbolicReasoningEnv:
    """完成“态势编码 → 事实生成 → 符号推理 → 可选执行”。"""

    def __init__(
        self,
        max_entities: int = 700,
        aim_tolerance_deg: float = 30.0,
    ) -> None:
        self.encoder = EntityEncoder(max_entities=max_entities)
        self.agent = SymbolicReasoningAgent()
        self.aim_tolerance_deg = aim_tolerance_deg
        self.current_step = 0

    def step(
        self,
        payload: Mapping[str, Any],
        execute_commands: bool = True,
        logger: Any = None,
        executor: Any = None,
    ) -> SymbolicStepResult:
        situation = self.encoder.encode(payload)
        facts = self.build_facts(situation)

        if execute_commands:
            run_result = self.agent.run(
                facts.values(),
                enemy_ids=[target.command_id for target in situation.targets],
                logger=logger,
                executor=executor,
            )
            decisions = run_result.decisions
            actions_dict = run_result.actions_dict
            execute_results = run_result.execute_results
            rewards = run_result.rewards
        else:
            decisions, actions_dict = self.agent.build_actions(facts.values())
            execute_results = None
            rewards = None

        self.current_step += 1
        return SymbolicStepResult(
            situation=situation,
            facts=facts,
            decisions=decisions,
            actions_dict=actions_dict,
            execute_results=execute_results,
            rewards=rewards,
        )

    def build_facts(
        self, situation: EncodedSituation
    ) -> Dict[str, ReasoningFacts]:
        facts_by_entity: Dict[str, ReasoningFacts] = {}
        targets = situation.targets

        for own in situation.own_entities:
            target = self._select_target(own, targets)
            if target is None:
                facts_by_entity[own.command_id] = ReasoningFacts(
                    entity_id=own.command_id
                )
                continue

            strike_range_km = own.strike_range_for(target)
            distance_km = own.distance_to_km(target)
            target_type_allowed = (
                target.domain is not TargetDomain.UNKNOWN and strike_range_km > 0.0
            )
            aimed = self._is_aimed(own, target)
            safety_clearance = own.communication_ok and not own.radar_jammed

            facts_by_entity[own.command_id] = ReasoningFacts(
                entity_id=own.command_id,
                target_id=target.command_id,
                # 新 SendMsg 没有剩余燃油字段，不能据此虚构返航事实。
                need_return_to_base=False,
                attack_authorized=own.commandable,
                target_type_allowed=target_type_allowed,
                # 新响应没有弹药余量；正打击距离表示该平台具备对应武器能力。
                weapon_available=strike_range_km > 0.0,
                within_attack_range=(
                    strike_range_km > 0.0 and distance_km <= strike_range_km
                ),
                aimed_at_target=aimed,
                safety_clearance=safety_clearance,
                chase_allowed=(
                    own.commandable and target_type_allowed and safety_clearance
                ),
                target_lon=target.longitude,
                target_lat=target.latitude,
            )

        return facts_by_entity

    @staticmethod
    def _select_target(
        own: EncodedEntity, targets: Sequence[EncodedEntity]
    ) -> Optional[EncodedEntity]:
        if not targets:
            return None
        capable_targets = [target for target in targets if own.strike_range_for(target) > 0]
        candidates = capable_targets or list(targets)
        return min(candidates, key=own.distance_to_km)

    def _is_aimed(self, own: EncodedEntity, target: EncodedEntity) -> bool:
        # 当前 actor_rules 同样对舰船/潜艇不要求机头对准。
        if own.domain is not TargetDomain.AIR:
            return True
        if own.heading_deg < 0.0:
            return False
        bearing = self._bearing_deg(
            own.longitude, own.latitude, target.longitude, target.latitude
        )
        difference = abs((own.heading_deg - bearing + 180.0) % 360.0 - 180.0)
        return difference <= self.aim_tolerance_deg

    @staticmethod
    def _bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
            math.radians, (lon1, lat1, lon2, lat2)
        )
        delta_lon = lon2_rad - lon1_rad
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (
            math.cos(lat1_rad) * math.sin(lat2_rad)
            - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        )
        return math.degrees(math.atan2(x, y)) % 360.0


def print_step(result: SymbolicStepResult, step_index: int) -> None:
    print(
        "step={} entities={} own={} targets={}".format(
            step_index,
            len(result.situation.entities),
            len(result.situation.own_entities),
            len(result.situation.targets),
        )
    )
    entities = {entity.command_id: entity for entity in result.situation.entities}
    for entity_id, decision in result.decisions.items():
        entity = entities.get(entity_id)
        name = entity.name if entity is not None else entity_id
        print("  {} -> {} | {}".format(name, decision.conclusion.value, decision.explanation))


def main_loop(
    env: SymbolicReasoningEnv,
    input_path: Path,
    steps: int,
    interval: float,
    execute_commands: bool,
    logger: Any,
) -> None:
    step_index = 0
    while steps <= 0 or step_index < steps:
        payload = load_situation(input_path)
        result = env.step(
            payload,
            execute_commands=execute_commands,
            logger=logger,
        )
        print_step(result, step_index)
        step_index += 1
        if steps <= 0 or step_index < steps:
            time.sleep(max(0.0, interval))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行符号推理测试智能体")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("source/我方视角下态势完整响应.txt"),
        help="SendMsg 完整响应 JSON 文件",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="循环次数；0 或负数表示持续运行",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="循环间隔秒数")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只编码和推理，不下发命令；默认会实际执行",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("symbolic_reasoning")
    env = SymbolicReasoningEnv()
    try:
        main_loop(
            env=env,
            input_path=args.input,
            steps=args.steps,
            interval=args.interval,
            execute_commands=not args.dry_run,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.info("符号推理循环已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
