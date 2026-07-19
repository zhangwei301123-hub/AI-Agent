"""前端 UI 推演状态监听与符号推理循环门控。"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional


ControlSignalProvider = Callable[[], Optional[str]]

# UI 的时间压缩编码仅用于显示和服务配置；符号推理规则始终按后台数据帧计时。
TIME_COMPRESSION_LABELS = {
    0: "1x",
    1: "2x",
    2: "5x",
    3: "10x",
    4: "15x",
    5: "30x",
    6: "40x",
    7: "50x",
    8: "60x",
    9: "Turbo",
}


def _load_project_signal_provider() -> ControlSignalProvider:
    # 根目录 execute.py 会初始化 gRPC，因此仅在启动监听线程时延迟导入。
    from execute import get_control_signal

    return get_control_signal


class FrontendControl:
    """轮询前端控制信号，并用 Event 控制符号推理是否继续运行。"""

    RUN_SIGNALS = frozenset(("start", "running", "restart", "resume"))

    def __init__(
        self,
        signal_provider: Optional[ControlSignalProvider] = None,
        poll_interval: float = 1.0,
        logger: Any = None,
    ) -> None:
        if poll_interval <= 0.0:
            raise ValueError("poll_interval 必须大于 0")
        self._signal_provider = signal_provider
        self.poll_interval = float(poll_interval)
        self.logger = logger
        self.control_event = threading.Event()
        self._listener_stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._state = "waiting"
        self._should_exit = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def should_exit(self) -> bool:
        with self._lock:
            return self._should_exit

    def start(self) -> None:
        """启动后台轮询；收到首个运行信号前保持暂停。"""

        if self._thread is not None and self._thread.is_alive():
            return
        if self._signal_provider is None:
            self._signal_provider = _load_project_signal_provider()
        self._thread = threading.Thread(
            target=self._listen,
            name="symbolic-reasoning-ui-control",
            daemon=True,
        )
        self._thread.start()

    def handle_signal(self, signal: Optional[str]) -> None:
        """处理 execute.get_control_signal() 返回的控制状态。"""

        normalized = str(signal or "").strip().lower()
        if normalized in self.RUN_SIGNALS:
            self._transition("running", should_exit=False, runnable=True)
        elif normalized == "pause":
            self._transition("paused", should_exit=False, runnable=False)
        elif normalized == "stop":
            self._transition("stopped", should_exit=True, runnable=False)
            # 唤醒正在 wait() 的主循环，使其能够检查 should_exit 并退出。
            self.control_event.set()
            self._listener_stop.set()
        else:
            self._log("warning", "[UI控制] 未知控制信号 %r，保持当前状态", signal)

    def handle_read_error(self, error: Exception) -> None:
        """读取前端失败时采用安全关闭策略，暂停推理和命令下发。"""

        was_error = self.state == "error"
        self._transition("error", should_exit=False, runnable=False)
        if not was_error:
            self._log(
                "warning",
                "[UI控制] 控制信号读取失败，符号推理保持暂停: %s",
                error,
            )

    def wait_until_runnable(self) -> bool:
        """阻塞到前端运行或停止；返回 False 表示主循环应退出。"""

        self.control_event.wait()
        return not self.should_exit and self.state == "running"

    def close(self, timeout: float = 2.0) -> None:
        self._listener_stop.set()
        self.control_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))

    def _listen(self) -> None:
        while not self._listener_stop.is_set():
            try:
                provider = self._signal_provider
                if provider is None:
                    raise RuntimeError("前端控制信号接口未初始化")
                self.handle_signal(provider())
            except Exception as error:  # gRPC 暂时不可用时继续轮询，以便自动恢复。
                self.handle_read_error(error)
            self._listener_stop.wait(self.poll_interval)

    def _transition(
        self,
        state: str,
        should_exit: bool,
        runnable: bool,
    ) -> None:
        with self._lock:
            previous = self._state
            self._state = state
            self._should_exit = should_exit
        if runnable:
            self.control_event.set()
        else:
            self.control_event.clear()
        if previous != state:
            self._log("info", "[UI控制] %s -> %s", previous, state)

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(message, *args)
