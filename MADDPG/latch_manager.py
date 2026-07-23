# latch_manager.py  「锁存表」
"""
对每个 <实体-ID, Actor-Index> 维护一个“冷却计时器”。
只要计时器 > 0，本轮就禁止覆盖该 Actor 的输出。
"""
import threading
import time
from collections import defaultdict
from .execute import *

DEFAULT_COOLDOWN = {
    0: 1200,   # AircraftTakeOffActor     （起飞全流程）
    1: 1200,   # ReturnToBaseActor
    2: 90,    # WayPointMove  （大机动）
    3: 30,    # Mobility      （小调速/升降）
    4: 10,    # AttackTarget  （一次武器发射序列）
    5: 180,    # SensorCtrl    （雷达/ECM 波束稳定）
    6: 360,   # DeploySonobuoyActor
    7: 3600,   # CancelAttackActor
    # 其余 Actor 若需要，也可自行添加
}
long_WayPointMove = 3600 * 5
class CommandLatch:
    """
    dict[entity_id][actor_idx] = remain_step
    remain_step>0 代表本帧禁止覆盖该 Actor。
    """
    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(int))
        # 维护一个字典 key是mdid  value 是每个actor_idx的剩余冷却时间
        self.running = threading.Event()
        self.running.set()
        self._lock = threading.Lock()
        self._stop_flag = False

    # ---------- 对外接口 ----------
    def tick(self):
        """每秒更新一次冷却时间"""
        # self.times = 0
        while not self._stop_flag:
            self.running.wait()
            with self._lock:
                for ent in list(self.table.keys()):# self.table.keys() mdID
                    for a_idx in list(self.table[ent].keys()):  #a_idx : actor_idx
                        v = self.table[ent][a_idx] = max(self.table[ent][a_idx]-TIME_SPEED_MAP[get_speed_rate()], 0)
                    # 可选：把全 0 的实体删掉，节省内存
                    if all(v == 0 for v in self.table[ent].values()):
                        self.table.pop(ent, None)
            time.sleep(1)  # 每秒更新一次
            # self.times += 1
            # print(f"[TICK] Tick thread running...{self.times}")

    def pause(self):
        self.running.clear()

    def resume(self):
        self.running.set()

    def stop(self):
        self._stop_flag = True
        self.running.set()

        
    def is_locked(self, ent_id, actor_idx) -> bool:
        with self._lock:
            return self.table[ent_id][actor_idx] > 0

    def lock(self, ent_id, actor_idx, cooldown=None):
        with self._lock:
            cd = cooldown if cooldown is not None else DEFAULT_COOLDOWN.get(actor_idx, 0)
            self.table[ent_id][actor_idx] = cd
            
class CommandLatch0:
    """
    dict[entity_id][actor_idx] = remain_step
    remain_step>0 代表本帧禁止覆盖该 Actor。
    """
    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(int))
        # 维护一个字典 key是mdid  value 是每个actor_idx的剩余冷却时间

    # ---------- 对外接口 ----------
    def tick(self):
        for ent in list(self.table.keys()):# self.table.keys() mdID
            for a_idx in list(self.table[ent].keys()):  #a_idx : actor_idx
                v = self.table[ent][a_idx] = max(self.table[ent][a_idx] - TIME_SPEED_MAP[get_speed_rate()], 0)
            # 可选：把全 0 的实体删掉，节省内存
            if all(v == 0 for v in self.table[ent].values()):
                self.table.pop(ent, None)

    def is_locked(self, ent_id, actor_idx) -> bool:
        return self.table[ent_id][actor_idx] > 0

    def lock(self, ent_id, actor_idx, cooldown=None):
        cd = cooldown if cooldown is not None else DEFAULT_COOLDOWN.get(actor_idx, 0)
        self.table[ent_id][actor_idx] = cd
        if actor_idx==2:
            cd = cooldown if cooldown is not None else DEFAULT_COOLDOWN.get(actor_idx+1, 0)
            self.table[ent_id][actor_idx+1] = cd





