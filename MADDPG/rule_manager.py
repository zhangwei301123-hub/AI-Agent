# import math
# from utils import geo_distance
# from engagement_rules import ENGAGEMENT_RULES, DEFAULT_COOLDOWN

# class EngagementRuleManager:
#     """
#     管理打击合法性判断规则，如攻击距离、类型限制、航向限制、冷却时间等。
#     """
#     def __init__(self, angle_threshold_deg=30, missile_range_factor=1842):
#         self.angle_threshold = math.radians(angle_threshold_deg)
#         self.range_factor = missile_range_factor

#     def can_attack(self, own_entity, target_entity, own_coords, tgt_coords) -> (bool, str):
#         """
#         判断 own_entity 是否可以攻击 target_entity

#         参数:
#         - own_entity: Tensor[30] 或 ndarray[30]，自身实体编码
#         - target_entity: Tensor[30] 或 ndarray[30]，目标实体编码
#         - own_coords: Tuple[float, float]，自身经纬度
#         - tgt_coords: Tuple[float, float]，目标经纬度

#         返回:
#         - (bool, str): 是否允许攻击 + 理由
#         """
#         own_type = int(own_entity[13])
#         own_alt  = int(own_entity[5])
#         tgt_type = int(target_entity[13])
#         tgt_alt  = int(target_entity[5])

#         # 规则1：实体类型合法
#         if own_type not in ENGAGEMENT_RULES.get(tgt_type, []):
#             return False, "类型不允许攻击"

#         # 规则2：距离合法
#         range_index = 29 if tgt_alt > 0 else 30 if tgt_alt < 0 else 31
#         attack_distance = geo_distance(*own_coords, *tgt_coords, alt_diff=0)
#         max_range = float(own_entity[range_index]) * self.range_factor
#         if attack_distance > max_range:
#             return False, "超出攻击距离"

#         # 规则3：航向合法（30°内）
#         cur_yaw_deg = float(own_entity[4])
#         cur_head = math.radians(cur_yaw_deg)
#         tgt_bearing = _bearing(*own_coords, *tgt_coords)
#         if abs(_rad_diff(cur_head, tgt_bearing)) > self.angle_threshold:
#             return False, "机头未对准"

#         return True, "允许攻击"

#     def adjust_to_maneuver(self, action_tensor, entity_idx, tgt_coords, alt_type=3):
#         """
#         将攻击改为机动动作（WayPointMove）

#         - action_tensor: Tensor[num_actor, E, 5]
#         - entity_idx: 当前实体在 action_tensor 中的索引
#         - tgt_coords: (dx, dy)，方向向量
#         - alt_type: 机动时的目标高度等级
#         """
#         wp = action_tensor[2][entity_idx]
#         wp[0] = 0.8
#         wp[1] = float(self._calculate_direction_index(*tgt_coords))
#         wp[2] = 0.0
#         wp[3] = 5.0
#         wp[4] = float(alt_type)

#     def cooldown_blocked(self, latch, ent_id: str, actor_idx: int) -> bool:
#         """
#         判断该实体的某个动作是否因为冷却时间被禁止执行。
#         """
#         return latch.is_locked(ent_id, actor_idx)

#     def _calculate_direction_index(self, dlon, dlat):
#         angle_rad = math.atan2(dlon, dlat)
#         if angle_rad < 0:
#             angle_rad += 2 * math.pi
#         return round(angle_rad / (math.pi / 4)) % 8
