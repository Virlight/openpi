"""
Core WebSocket server implementation with readable action debug logging.

Protocol with robot_client:
1. Send metadata immediately after connection (msgpack).
2. Receive observation messages from client (msgpack).
3. Run policy.infer(obs) and return result (msgpack).
4. Send text error message when inference fails.
"""

import json
import logging
import os
import sys
import threading
from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# ANSI colour helpers – only active when stderr is a real terminal.
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None

class _C:
    RESET  = "\033[0m"   if _USE_COLOR else ""
    GREY   = "\033[90m"  if _USE_COLOR else ""   # step prefix
    CYAN   = "\033[96m"  if _USE_COLOR else ""   # key names
    YELLOW = "\033[93m"  if _USE_COLOR else ""   # numeric values (non-gripper)
    GREEN  = "\033[92m"  if _USE_COLOR else ""   # gripper joint angle in rad (index 6)
    DIM    = "\033[2m"   if _USE_COLOR else ""   # separators


def _format_action_vec(vals: list) -> str:
    """Format a 7D action vector, colouring gripper (index 6) in green."""
    if len(vals) != 7:
        return f"{_C.YELLOW}{json.dumps(vals, ensure_ascii=False)}{_C.RESET}"
    body = ", ".join(
        (f"{_C.GREEN}{v}{_C.RESET}" if i == 6 else f"{_C.YELLOW}{v}{_C.RESET}")
        for i, v in enumerate(vals)
    )
    return f"[{body}]"

try:
    import msgpack
except ImportError:
    raise ImportError("Please install: pip install msgpack")

# msgpack-numpy is recommended (client may send numpy arrays), but keep it optional:
# if not installed, numpy arrays will arrive as a dict (data/shape) and can be handled in convert_input/utils.
try:
    import msgpack_numpy as m  # type: ignore
    m.patch()
except ImportError:
    m = None

try:
    import websockets
    import websockets.sync.server
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError("Please install: pip install websockets")

logger = logging.getLogger(__name__)


def _to_debug_jsonable(value: Any) -> Any:
    """Convert payload values into JSON-serializable structures for DEBUG logs."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_debug_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_debug_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return value


def _action_payload_for_debug(result: Any, max_steps: int = 4) -> str:
    """Extract the outgoing action payload for readable DEBUG logs.

    For multi-step action tensors (shape [T, ...]), only the first *max_steps*
    timesteps are shown, one line per timestep:
        step 0 | follow1_pos=[...] follow2_pos=[...]
        step 1 | ...
    """
    if not isinstance(result, dict):
        return json.dumps(_to_debug_jsonable(result), ensure_ascii=False)

    action_keys = (
        "follow1_pos",
        "follow2_pos",
        "follow1_joints",
        "follow2_joints",
    )
    present = [k for k in action_keys if k in result]
    if not present:
        return json.dumps(_to_debug_jsonable(result), ensure_ascii=False)

    # Determine number of timesteps from the first present key.
    first_val = result[present[0]]
    if isinstance(first_val, np.ndarray):
        n_steps = first_val.shape[0]
    elif isinstance(first_val, (list, tuple)):
        n_steps = len(first_val)
    else:
        # Scalar / unknown — fall back to flat dump.
        payload = {k: _to_debug_jsonable(result[k]) for k in present}
        return json.dumps(payload, ensure_ascii=False)

    lines = []
    for t in range(min(n_steps, max_steps)):
        parts = []
        for k in present:
            v = result[k]
            if isinstance(v, np.ndarray):
                step_val = np.round(v[t], 4).tolist()
            elif isinstance(v, (list, tuple)):
                step_val = _to_debug_jsonable(v[t])
            else:
                step_val = _to_debug_jsonable(v)
            val_str = (
                _format_action_vec(step_val)
                if isinstance(step_val, list) and len(step_val) == 7
                else f"{_C.YELLOW}{json.dumps(step_val, ensure_ascii=False)}{_C.RESET}"
            )
            parts.append(f"{_C.CYAN}{k}{_C.RESET}{_C.DIM}={_C.RESET}{val_str}")
        prefix = f"{_C.GREY}  step {t:02d} |{_C.RESET} "
        lines.append(prefix + f"  {_C.DIM}|{_C.RESET}  ".join(parts))

    suffix = (
        f"  {_C.GREY}... ({n_steps - max_steps} more steps){_C.RESET}"
        if n_steps > max_steps else ""
    )
    return "\n" + "\n".join(lines) + suffix


class WebSocketModelServer:
    """WebSocket server that handles robot_client connections."""
    
    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """
        Initialize server.
        
        Args:
            policy: Policy object with infer() method and metadata property.
            host: Server host.
            port: Server port.
        """
        self.policy = policy
        self.host = host
        self.port = port
        self._infer_lock = threading.Lock()

    def _reset_policy(self, reason: str) -> None:
        if not hasattr(self.policy, "reset"):
            return
        try:
            self.policy.reset()
            logger.info("Policy reset (%s)", reason)
        except Exception:
            logger.exception("Policy reset failed (%s)", reason)
    
    def _handle_client(self, conn: websockets.sync.server.ServerConnection) -> None:
        """Handle a single client connection."""
        client_addr = conn.remote_address
        logger.info(f"Client connected: {client_addr}")
        self._reset_policy("client_connected")
        
        try:
            # 1. Send metadata right after connection
            metadata = getattr(self.policy, "metadata", {}) or {}
            # Use explicit msgpack options to avoid bytes keys/values surprises across environments.
            metadata_bytes = msgpack.packb(metadata, use_bin_type=True)
            conn.send(metadata_bytes)
            logger.info(f"Sent metadata to {client_addr}: {metadata}")
            
            # 2. Handle inference requests in a loop
            while True:
                try:
                    # Receive observation
                    message = conn.recv()
                    
                    # If this is a text message (error/control), log and continue
                    if isinstance(message, str):
                        logger.warning(f"Received text message from {client_addr}: {message}")
                        continue
                    
                    # Decode observation
                    # raw=False ensures str keys (compatible with robot_client's dict access patterns).
                    obs = msgpack.unpackb(message, raw=False)
                    logger.debug(f"Received observation from {client_addr}, keys: {list(obs.keys())}")
                    
                    # Run policy inference
                    try:
                        with self._infer_lock:
                            result = self.policy.infer(obs)
                    except Exception as exc:
                        logger.exception(f"Policy inference error for {client_addr}")
                        # Send error message as text
                        conn.send(f"Error in policy inference: {exc}", text=True)
                        continue
                    
                    # Encode and send result
                    result_bytes = msgpack.packb(result, use_bin_type=True)
                    logger.debug("Sending action payload to %s: %s", client_addr, _action_payload_for_debug(result))
                    conn.send(result_bytes)
                    logger.debug("Sent result to %s (%d bytes)", client_addr, len(result_bytes))
                    
                except ConnectionClosed:
                    logger.info(f"Client disconnected: {client_addr}")
                    break
                    
        except Exception:
            logger.exception(f"Unhandled error in client handler for {client_addr}")
        finally:
            self._reset_policy("client_disconnected")
            logger.info(f"Client handler finished: {client_addr}")
    
    def serve_forever(self) -> None:
        """Start server and run forever."""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Starting WebSocket server on {uri}")
        
        with websockets.sync.server.serve(
            self._handle_client,
            host=self.host,
            port=self.port,
            max_size=None,  # No message size limit
            compression=None,  # Disable compression for performance
        ) as server:
            logger.info("=" * 60)
            logger.info(f"WebSocket server is running on {uri}")
            logger.info("Server is ready and waiting for connections...")
            logger.info("=" * 60)
            server.serve_forever()
