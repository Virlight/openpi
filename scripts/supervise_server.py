#!/usr/bin/env python3
"""Restart a long-running server command if it exits.

Usage:
    python scripts/supervise_server.py -- python serve.py --checkpoint /path/to/ckpt

The supervisor exits only when it receives Ctrl+C/SIGTERM, or when
--max-restarts is reached.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence


_stop_requested = False
_child: subprocess.Popen[object] | None = None


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - supervisor - {message}", flush=True)


def _request_stop(signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True
    _log(f"received signal {signum}; stopping child and exiting")
    _stop_child()


def _stop_child() -> None:
    child = _child
    if child is None or child.poll() is not None:
        return

    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return

    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        _log("child did not stop after SIGINT; sending SIGTERM")
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _log("child did not stop after SIGTERM; sending SIGKILL")
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _log("child still did not stop after SIGKILL")


def _check_websocket_health(uri: str, timeout: float) -> tuple[bool, str]:
    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        ok, message = _check_tcp_health(uri, timeout)
        if ok:
            return True, f"tcp ok; websocket health unavailable: {exc}"
        return False, f"websocket unavailable ({exc}); tcp check failed: {message}"

    try:
        with connect(
            uri,
            open_timeout=timeout,
            close_timeout=1,
            max_size=None,
            compression=None,
        ) as conn:
            first = conn.recv(timeout=timeout)
            if isinstance(first, str):
                return False, "first frame was text; expected metadata bytes"
            return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _check_tcp_health(uri: str, timeout: float) -> tuple[bool, str]:
    try:
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        host = parsed.hostname
        if not host:
            return False, f"cannot parse host from {uri!r}"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep a server command alive by restarting it after exits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=5.0,
        help="Seconds to wait before restarting after a normal-length run.",
    )
    parser.add_argument(
        "--min-run-seconds",
        type=float,
        default=30.0,
        help="Runs shorter than this are treated as crash loops and use backoff.",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=120.0,
        help="Maximum restart delay while crash-loop backoff is active.",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        help="Maximum restarts; 0 means unlimited.",
    )
    parser.add_argument(
        "--health-uri",
        default="",
        help="Optional WebSocket URI to probe, e.g. ws://127.0.0.1:8000. Empty disables health checks.",
    )
    parser.add_argument(
        "--health-start-period",
        type=float,
        default=180.0,
        help="Seconds to wait after each child start before the first health check.",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=30.0,
        help="Seconds between health checks when --health-uri is set.",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for the health WebSocket handshake and metadata receive.",
    )
    parser.add_argument(
        "--health-failures",
        type=int,
        default=3,
        help="Consecutive failed health checks before restarting the child.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Put it after '--'.",
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command; example: python scripts/supervise_server.py -- python serve.py --checkpoint CKPT")
    if args.restart_delay < 0:
        parser.error("--restart-delay must be >= 0")
    if args.min_run_seconds < 0:
        parser.error("--min-run-seconds must be >= 0")
    if args.max_backoff < args.restart_delay:
        parser.error("--max-backoff must be >= --restart-delay")
    if args.max_restarts < 0:
        parser.error("--max-restarts must be >= 0")
    if args.health_start_period < 0:
        parser.error("--health-start-period must be >= 0")
    if args.health_interval <= 0:
        parser.error("--health-interval must be > 0")
    if args.health_timeout <= 0:
        parser.error("--health-timeout must be > 0")
    if args.health_failures <= 0:
        parser.error("--health-failures must be > 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    global _child

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    restarts = 0
    backoff = float(args.restart_delay)

    while not _stop_requested:
        started_at = time.monotonic()
        health_failures = 0
        next_health_at = started_at + float(args.health_start_period)
        _log("starting child: " + " ".join(args.command))
        _child = subprocess.Popen(args.command, start_new_session=True)
        return_code: int | None = None
        while not _stop_requested:
            try:
                return_code = _child.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass

            if not args.health_uri or time.monotonic() < next_health_at:
                continue

            next_health_at = time.monotonic() + float(args.health_interval)
            ok, message = _check_websocket_health(
                args.health_uri,
                float(args.health_timeout),
            )
            if ok:
                if health_failures:
                    _log("health check recovered")
                health_failures = 0
                continue

            health_failures += 1
            _log(
                f"health check failed ({health_failures}/{args.health_failures}): "
                f"{message}"
            )
            if health_failures >= args.health_failures:
                _log("health check failure threshold reached; restarting child")
                _stop_child()
                return_code = _child.poll()
                if return_code is None:
                    return_code = _child.wait()
                break

        _child = None

        runtime = time.monotonic() - started_at
        if _stop_requested:
            break

        restarts += 1
        _log(
            f"child exited with code {return_code} after {runtime:.1f}s; "
            f"restart #{restarts}"
        )

        if args.max_restarts and restarts >= args.max_restarts:
            _log("--max-restarts reached; exiting")
            return return_code

        if runtime >= args.min_run_seconds:
            backoff = float(args.restart_delay)
        else:
            backoff = min(max(backoff * 2.0, args.restart_delay), args.max_backoff)

        if backoff > 0:
            _log(f"restarting in {backoff:.1f}s")
            deadline = time.monotonic() + backoff
            while not _stop_requested and time.monotonic() < deadline:
                time.sleep(min(0.5, deadline - time.monotonic()))

    _log("supervisor stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
