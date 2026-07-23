import time
import threading
from collections import defaultdict
from datetime import datetime

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

class CommandLatch:
    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(int))
        self.running = threading.Event()
        self.running.set()
        self._lock = threading.Lock()
        self._stop_flag = False

    def tick(self):
        self.times = 0
        while not self._stop_flag:
            self.running.wait()
            with self._lock:
                for ent in list(self.table.keys()):
                    for a_idx in list(self.table[ent].keys()):
                        self.table[ent][a_idx] = max(self.table[ent][a_idx] - 1, 0)
                    if all(v == 0 for v in self.table[ent].values()):
                        del self.table[ent]
            time.sleep(1)
            self.times += 1
            log(f"[TICK] Tick thread running...{self.times}")

    def pause(self):
        self.running.clear()

    def resume(self):
        self.running.set()

    def stop(self):
        self._stop_flag = True
        self.running.set()

    def is_locked(self, ent_id, actor_idx):
        with self._lock:
            return self.table[ent_id][actor_idx] > 0

    def lock(self, ent_id, actor_idx, cooldown):
        with self._lock:
            self.table[ent_id][actor_idx] = cooldown

class SimulatedEnv:
    def __init__(self):
        self.latch = None
        self.tick_thread = None

    def reset(self):
        log("[CONTROL] Resetting environment...")
        self.stop()
        self.latch = CommandLatch()
        self.tick_thread = threading.Thread(target=self.latch.tick, daemon=True)
        self.tick_thread.start()

    def pause(self):
        log("[CONTROL] Pausing tick...")
        if self.latch:
            self.latch.pause()

    def resume(self):
        log("[CONTROL] Resuming tick...")
        if self.latch:
            self.latch.resume()

    def stop(self):
        log("[CONTROL] Stopping tick...")
        if self.latch:
            self.latch.stop()
        if self.tick_thread:
            self.tick_thread.join()

    def step(self, ent_id, actor_idx, cooldown=3):
        if not self.latch or not self.latch.running.is_set():
            log("[STEP] Environment paused or not ready.")
            return False
        if not self.latch.is_locked(ent_id, actor_idx):
            log(f"[STEP] Entity {ent_id} Actor {actor_idx} is not locked. Executing action.")
            self.latch.lock(ent_id, actor_idx, cooldown)
            return True
        else:
            log(f"[STEP] Entity {ent_id} Actor {actor_idx} is locked. Cannot execute action.")
            return False

class SignalListener:
    def __init__(self, env: SimulatedEnv):
        self.env = env
        self.state = "idle"
        self.control_event = threading.Event()
        self.simulating = False

    def handle_signal(self, sig):
        if sig == "start":
            if self.state in ["idle", "stopped"]:
                self.env.reset()
                self.simulating = True
                self.control_event.set()
            elif self.state == "paused":
                self.env.resume()
                self.control_event.set()
            self.state = "running"

        elif sig == "pause":
            self.env.pause()
            self.control_event.clear()
            self.state = "paused"

        elif sig == "stop":
            self.env.stop()
            self.control_event.clear()
            self.simulating = False
            self.state = "stopped"

        elif sig == "restart":
            self.env.stop()
            self.env.reset()
            self.simulating = True
            self.control_event.set()
            self.state = "running"

        log(f"[SIGNAL] Received signal: {sig} → new state: {self.state}")

    def simulate_signals(self):
        signals = ["start", "pause", "start", "stop", "restart", "stop"]
        for sig in signals:
            time.sleep(5)
            self.handle_signal(sig)

def main_loop(env: SimulatedEnv, controller: SignalListener):
    step_count = 0
    log("[MAIN] Waiting for 'start' signal...")
    while controller.simulating:
        controller.control_event.wait()  # 阻塞直到 start/resume
        if not controller.simulating:
            break

        success = env.step(ent_id=1, actor_idx=1, cooldown=3)
        if success:
            step_count += 1
            log(f"[MAIN] Step {step_count} executed.")
        else:
            log(f"[MAIN] Step {step_count + 1} skipped (locked or paused).")

        if step_count >= 50:
            log("[MAIN] Reached max step limit. Ending this simulation.")
            controller.simulating = False
            break

        time.sleep(0.5)

def main():
    env = SimulatedEnv()
    controller = SignalListener(env)

    threading.Thread(target=controller.simulate_signals, daemon=True).start()

    while True:
        # 等待 signal 设置 controller.simulating=True
        while not controller.simulating:
            time.sleep(0.2)

        # 开始执行主循环（可中途被 stop/restart）
        main_loop(env, controller)

        log("[MAIN] Simulation ended. Waiting for new signal...")

if __name__ == "__main__":
    main()
