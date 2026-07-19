from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from symbolic_reasoning.control import FrontendControl
from symbolic_reasoning.symbolic_reasoning4test import (
    SymbolicReasoningEnv,
    main_loop,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "source" / "我方视角下态势完整响应.txt"


class CapturingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else str(message))

    def warning(self, message, *args):
        self.messages.append(message % args if args else str(message))


class FrontendControlTests(unittest.TestCase):
    def test_listener_thread_polls_provider_and_reaches_running(self):
        signals = ["pause", "running"]

        def provider():
            return signals.pop(0) if signals else "running"

        control = FrontendControl(
            signal_provider=provider,
            poll_interval=0.01,
        )
        control.start()
        try:
            self.assertTrue(control.control_event.wait(timeout=1.0))
            self.assertEqual(control.state, "running")
        finally:
            control.close()

    def test_symbolic_main_loop_pauses_and_resumes_with_ui_signal(self):
        logger = CapturingLogger()
        control = FrontendControl(
            signal_provider=lambda: "pause",
            poll_interval=0.01,
            logger=logger,
        )
        control.handle_signal("pause")
        env = SymbolicReasoningEnv(max_entities=64)
        worker = threading.Thread(
            target=main_loop,
            kwargs={
                "env": env,
                "input_path": SAMPLE,
                "steps": 1,
                "interval": 0.0,
                "execute_commands": False,
                "logger": logger,
                "frontend_control": control,
            },
        )

        worker.start()
        time.sleep(0.03)
        self.assertTrue(worker.is_alive())
        self.assertEqual(env.current_step, 0)

        control.handle_signal("running")
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(env.current_step, 1)
        self.assertIn("推理路径：", "\n".join(logger.messages))

    def test_pause_blocks_until_running_signal_resumes(self):
        logger = CapturingLogger()
        control = FrontendControl(
            signal_provider=lambda: "pause",
            poll_interval=0.01,
            logger=logger,
        )
        control.handle_signal("pause")
        result = []
        waiter = threading.Thread(
            target=lambda: result.append(control.wait_until_runnable())
        )

        waiter.start()
        time.sleep(0.03)
        self.assertTrue(waiter.is_alive())

        # execute.py 将协议 code=3 转成 running；该信号也必须恢复执行。
        control.handle_signal("running")
        waiter.join(timeout=1.0)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [True])
        self.assertEqual(control.state, "running")
        self.assertIn("paused -> running", "\n".join(logger.messages))

    def test_stop_wakes_paused_waiter_and_requests_exit(self):
        control = FrontendControl(
            signal_provider=lambda: "pause",
            poll_interval=0.01,
        )
        control.handle_signal("pause")
        result = []
        waiter = threading.Thread(
            target=lambda: result.append(control.wait_until_runnable())
        )
        waiter.start()
        time.sleep(0.03)

        control.handle_signal("stop")
        waiter.join(timeout=1.0)

        self.assertEqual(result, [False])
        self.assertTrue(control.should_exit)
        self.assertEqual(control.state, "stopped")

    def test_signal_read_error_fails_closed_until_recovery(self):
        control = FrontendControl(
            signal_provider=lambda: "running",
            poll_interval=0.01,
        )
        control.handle_signal("running")
        self.assertTrue(control.control_event.is_set())

        control.handle_read_error(RuntimeError("offline"))
        self.assertFalse(control.control_event.is_set())
        self.assertEqual(control.state, "error")

        control.handle_signal("start")
        self.assertTrue(control.control_event.is_set())
        self.assertEqual(control.state, "running")


if __name__ == "__main__":
    unittest.main()
