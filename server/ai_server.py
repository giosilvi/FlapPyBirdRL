"""
Flappy Bird AI Web Server

A FastAPI server that runs a trained DQN agent playing Flappy Bird and streams
the game state to browser clients via WebSocket. The browser renders the game
using HTML5 Canvas with the original game assets.

Usage:
    uvicorn server.ai_server:app --reload --port 8765

Then open http://localhost:8765 in your browser.
"""
import asyncio
import json
import os
import sys
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Set, Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rl.dqn_agent import DQNAgent, DQNConfig
from rl.pipe_controller_agent import PipeControllerAgent, PipeControllerConfig
from src.ai_env import FlappyEnv

app = FastAPI()

# Mount assets directory
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Mount static files directory (for CSS)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global state
env = None
agent = None
pipe_controllers: Optional[PipeControllerAgent] = None
clients: Set[WebSocket] = set()
current_checkpoint_path: Optional[str] = None  # Track currently loaded checkpoint

# Experience buffers for online learning
bird_experiences: deque = deque(maxlen=10000)  # Store recent experiences for bird DQN
pipe_experiences: deque = deque(maxlen=10000)  # Store recent experiences for pipe controllers (deprecated - now pushed on death)
current_game_pipe_experiences: list = []  # Store pipe experiences for current game (pushed to buffer on death)
training_step_counter = 0

# ---- Multiplayer in-memory state (simple, single-process) ----
users: Dict[
    str, Dict
] = (
    {}
)  # userId -> { 'ws': WebSocket, 'joined': bool, 'assigned': int|None, 'last_dy': float, 'color': str }
user_queue: list[str] = []  # userIds waiting for assignment
next_pair_id: int = 1

# Color palette for up to 5 players (high contrast)
PLAYER_COLORS = [
    "#00baff",  # electric blue
    "#ff4d4d",  # red
    "#06d6a0",  # green
    "#ffd166",  # yellow
    "#9b5de5",  # purple
]


def _get_or_assign_color_for_user(uid: str) -> str:
    # return existing
    if uid in users and users[uid].get("color"):
        return users[uid]["color"]
    # choose first unused among joined users; else first
    used = {u.get("color") for u in users.values() if u.get("joined")}
    color = next((c for c in PLAYER_COLORS if c not in used), PLAYER_COLORS[0])
    users.setdefault(uid, {})["color"] = color
    return color


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------- Small helper utilities ------------------------- #


def can_spawn_pipes_modified_impl(env) -> bool:
    """Return True when we want the env to spawn more pipes for wide view."""
    last = env.pipes.upper[-1] if env.pipes.upper else None
    if not last:
        return True
    return last.x < 1300


def remove_old_pipes_modified_impl(
    env, users: Dict[str, Dict], user_queue: list[str]
) -> None:
    """Remove old pipes and requeue owners for reassignment."""
    for up, low in list(zip(env.pipes.upper, env.pipes.lower)):
        if up.x < -up.w - 200:
            owner = getattr(up, "ownerUserId", None) or getattr(
                low, "ownerUserId", None
            )
            if owner and owner in users:
                users[owner]["assigned"] = None
                user_queue.append(owner)
            try:
                env.pipes.upper.remove(up)
            except ValueError:
                pass
            try:
                env.pipes.lower.remove(low)
            except ValueError:
                pass


def _extract_pipe_controller_state(env, pipe_idx: int) -> Optional[np.ndarray]:
    """
    Extract state for pipe controller at given index.
    State: [dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y]
    
    Args:
        env: Environment
        pipe_idx: Index of pipe pair (0-3 for next 4 pipes, 0 is closest to bird)
        
    Returns:
        State array or None if pipe doesn't exist
    """
    if pipe_idx >= len(env.pipes.upper):
        return None
    
    up = env.pipes.upper[pipe_idx]
    low = env.pipes.lower[pipe_idx]
    
    # Calculate dx: normalized horizontal distance
    dx = (up.x - env.player.x) / float(env.window.width)
    
    # Calculate dy: normalized vertical distance to gap center
    gap_center_y = low.y - env.pipes.pipe_gap / 2.0
    dy = (gap_center_y - env.player.y) / float(env.window.viewport_height)
    
    # Bird y position (normalized)
    bird_y = env.player.y / float(env.window.viewport_height)
    
    # Bird velocity (normalized)
    vy_cap = max(abs(env.player.max_vel_y), abs(env.player.min_vel_y))
    bird_vel_y = np.clip(env.player.vel_y / float(vy_cap), -1.0, 1.0)
    
    # Gap center y position of the pipe AHEAD (closer to bird) - for coordination
    # pipe_idx 0 is closest to bird, so pipe_idx-1 is ahead (if exists)
    ahead_pipe_gap_center_y_norm = 0.5  # Default to center if no pipe ahead
    if pipe_idx > 0 and (pipe_idx - 1) < len(env.pipes.lower):
        ahead_low = env.pipes.lower[pipe_idx - 1]
        ahead_gap_center_y = ahead_low.y - env.pipes.pipe_gap / 2.0
        ahead_pipe_gap_center_y_norm = ahead_gap_center_y / float(env.window.viewport_height)
    
    return np.array([dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y_norm], dtype=np.float32)


def apply_user_inputs_to_pipes(env, users: Dict[str, Dict]) -> None:
    """
    Apply per-user dy input to their owned pipe pair, with bounds checking.
    If pipe has no user owner, use NN controller instead.
    """
    _ensure_pair_metadata(env)
    pipe_gap = env.pipes.pipe_gap
    viewport_h = env.window.viewport_height
    pipe_h = env.images.pipe[0].get_height()
    
    # Track which pipes are controlled by NNs (for experience collection)
    nn_controlled_pipes = {}
    
    # Debug: Check if pipe_controllers is initialized (only log once per call)
    if pipe_controllers is None and not hasattr(apply_user_inputs_to_pipes, "_logged_no_controllers"):
        print("⚠ WARNING: pipe_controllers is None - pipes won't move!", flush=True)
        setattr(apply_user_inputs_to_pipes, "_logged_no_controllers", True)
    
    for idx, (up, low) in enumerate(zip(env.pipes.upper, env.pipes.lower)):
        owner = getattr(up, "ownerUserId", None)
        dy = 0.0
        
        if owner:
            # User control takes priority
            u = users.get(owner)
            if u:
                dy = float(u.get("last_dy", 0.0))
        elif pipe_controllers is not None and idx < 4:
            # Use NN controller for first 4 pipes (if no user control)
            state = _extract_pipe_controller_state(env, idx)
            if state is not None:
                dy = pipe_controllers.act(idx, state)
                nn_controlled_pipes[idx] = {"state": state, "action": dy}
            else:
                # State extraction failed - this shouldn't happen normally
                dy = 0.0
        
        if abs(dy) < 1e-6:
            continue
        
        center = low.y - pipe_gap / 2.0
        # Apply movement: both user and NN use 3.0 multiplier (consistent with training)
        movement_multiplier = 3.0  # Same speed for user and NN control
        center += dy * movement_multiplier
        min_center = pipe_gap / 2.0 + 5
        max_center = viewport_h - pipe_gap / 2.0 - 5
        if center < min_center:
            center = min_center
        if center > max_center:
            center = max_center
        up.y = center - pipe_gap / 2.0 - pipe_h
        low.y = center + pipe_gap / 2.0
    
    # Store NN-controlled pipe info for experience collection
    if nn_controlled_pipes:
        setattr(apply_user_inputs_to_pipes, "_last_nn_pipes", nn_controlled_pipes)


def release_offscreen_owned(
    env, users: Dict[str, Dict], user_queue: list[str]
) -> None:
    """Release ownership for any pair that moved left of the screen and requeue."""
    for up, low in zip(env.pipes.upper, env.pipes.lower):
        owner = getattr(up, "ownerUserId", None)
        if owner and up.x < 0:
            setattr(up, "ownerUserId", None)
            setattr(low, "ownerUserId", None)
            if owner in users:
                users[owner]["assigned"] = None
                if owner not in user_queue:
                    user_queue.append(owner)


def broadcast_frame_state(a: int, scored: bool, state: dict, checkpoint_name: Optional[str] = None) -> dict:
    """Build the broadcast payload for a frame update."""
    payload = {
        "type": "frame",
        "state": state,
        "colors": {
            uid: u.get("color") for uid, u in users.items() if u.get("color")
        },
        "events": {"flap": a == 1, "score": scored},
    }
    if checkpoint_name:
        payload["checkpoint"] = checkpoint_name
    return payload


def _select_action(agent: DQNAgent, s) -> int:
    with torch.no_grad():
        state_t = torch.from_numpy(s).float().unsqueeze(0).to(agent.device)
        q_vals = agent.q(state_t)
        return int(torch.argmax(q_vals, dim=-1).item())


def _after_step_maintenance(env) -> None:
    _fill_wide_view_pipes(env, target_last_x=2100, spacing=180)
    apply_user_inputs_to_pipes(env, users)
    release_offscreen_owned(env, users, user_queue)
    _assign_waiting_users_to_pipes(env)


def _find_latest_checkpoint() -> Optional[Path]:
    """Find the latest bird checkpoint from training or web evaluation"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return None
    
    # Priority order:
    # 1. best_online.pt (from web evaluation auto-training)
    # 2. Step-based checkpoints (from main training)
    # 3. best.pt (fallback)
    
    best_online_pt = checkpoints_dir / "best_online.pt"
    if best_online_pt.exists():
        return best_online_pt
    
    # Look for step-based checkpoints (from training)
    step_ckpts = list(checkpoints_dir.glob("bird_ckpt_step_*.pt"))
    if step_ckpts:
        # Sort by modification time (newest first)
        latest = max(step_ckpts, key=lambda p: p.stat().st_mtime)
        return latest
    
    # Fallback to best.pt if no step checkpoints exist
    best_pt = checkpoints_dir / "best.pt"
    if best_pt.exists():
        return best_pt
    
    return None


def _reload_latest_checkpoint_if_newer() -> bool:
    """Reload agent and pipe controllers if newer checkpoints are available. Returns True if reloaded."""
    global agent, pipe_controllers, current_checkpoint_path
    
    if agent is None:
        return False
    
    latest_ckpt = _find_latest_checkpoint()
    if latest_ckpt is None:
        return False
    
    latest_path = str(latest_ckpt)
    
    # If this is the same checkpoint we already have loaded, skip
    if current_checkpoint_path == latest_path:
        return False
    
    # Check if latest checkpoint is newer than current
    if current_checkpoint_path:
        current_path = Path(current_checkpoint_path)
        if current_path.exists():
            current_mtime = current_path.stat().st_mtime
            latest_mtime = latest_ckpt.stat().st_mtime
            
            if latest_mtime <= current_mtime:
                return False  # No newer checkpoint
    
    # Reload agent with latest checkpoint
    try:
        device = get_device()
        state_dim = 8
        action_dim = 2
        cfg = DQNConfig(state_dim=state_dim, action_dim=action_dim, device=device)
        agent = DQNAgent(cfg)
        agent.load(latest_path)
        current_checkpoint_path = latest_path  # Update tracked checkpoint
        print(f"✓ Reloaded agent from latest checkpoint: {latest_ckpt.name}")
        
        # Also reload pipe controllers if they exist (they're saved together during training)
        pipe_checkpoint_dir = Path("checkpoints/pipe_controllers")
        if pipe_checkpoint_dir.exists():
            # Initialize pipe controllers if they don't exist
            if pipe_controllers is None:
                pipe_cfg = PipeControllerConfig(
                    state_dim=5,
                    device=device,
                    num_controllers=4,
                )
                pipe_controllers = PipeControllerAgent(pipe_cfg)
                print(f"✓ Initialized pipe controllers (were None)")
            
            try:
                pipe_controllers.load(str(pipe_checkpoint_dir))
                # Clear the replay buffer for web evaluation (we want to fill it from deaths)
                pipe_controllers.replay_buffer["pos"] = 0
                pipe_controllers.replay_buffer["full"] = False
                print(f"✓ Reloaded pipe controllers from {pipe_checkpoint_dir}")
                print(f"✓ Replay buffer cleared (will fill from game deaths)")
            except Exception as e:
                print(f"⚠ Error reloading pipe controllers: {e}")
        
        return True
    except Exception as e:
        print(f"⚠ Error reloading checkpoint {latest_path}: {e}")
        return False


def _on_game_over(env) -> None:
    _fill_wide_view_pipes(env, target_last_x=2000, spacing=180)
    user_queue.clear()
    for uid, u in users.items():
        u["joined"] = False
        u["assigned"] = None
        u["last_dy"] = 0.0
    for up, low in zip(env.pipes.upper, env.pipes.lower):
        setattr(up, "ownerUserId", None)
        setattr(low, "ownerUserId", None)
    
    # Reload latest checkpoint if available
    _reload_latest_checkpoint_if_newer()


def _ws_send_init(websocket: WebSocket) -> None:
    checkpoint_name = None
    if current_checkpoint_path:
        checkpoint_name = Path(current_checkpoint_path).name
    
    payload = {
        "type": "init",
        "state": extract_game_state(),
        "colors": {
            uid: u.get("color")
            for uid, u in users.items()
            if u.get("color")
        },
        "auto_training_rounds": auto_training_rounds_completed,
    }
    if checkpoint_name:
        payload["checkpoint"] = checkpoint_name
    asyncio.create_task(websocket.send_json(payload))


def _handle_join_request(websocket: WebSocket) -> None:
    joined_count = sum(1 for u in users.values() if u.get("joined"))
    if joined_count >= len(PLAYER_COLORS):
        asyncio.create_task(
            websocket.send_json({"type": "join_denied", "reason": "full"})
        )
        return
    uid = str(uuid.uuid4())
    color = _get_or_assign_color_for_user(uid)
    users[uid] = {
        "ws": websocket,
        "joined": True,
        "assigned": None,
        "last_dy": 0.0,
        "color": color,
    }
    user_queue.append(uid)
    asyncio.create_task(
        websocket.send_json({"type": "join_ok", "userId": uid, "color": color})
    )


def _handle_leave(uid: str) -> None:
    users[uid]["joined"] = False
    users[uid]["last_dy"] = 0.0
    assigned = users[uid].get("assigned")
    if assigned is not None:
        for up, low in zip(env.pipes.upper, env.pipes.lower):
            if getattr(up, "pair_id", None) == assigned:
                setattr(up, "ownerUserId", None)
                setattr(low, "ownerUserId", None)
                break
        users[uid]["assigned"] = None
    try:
        user_queue.remove(uid)
    except ValueError:
        pass


def _handle_input(uid: str, dy_val) -> None:
    dy = float(dy_val)
    dy = -1.0 if dy < -1 else (1.0 if dy > 1 else dy)
    users[uid]["last_dy"] = dy


def process_ws_message(msg: dict, websocket: WebSocket) -> None:
    """Handle a single websocket message. Mutates global state."""
    typ = msg.get("type")
    if typ == "join_request":
        _handle_join_request(websocket)
        return

    uid = msg.get("userId")
    if not uid or uid not in users:
        return
    users[uid]["ws"] = websocket
    if typ == "leave":
        _handle_leave(uid)
    elif typ == "input":
        _handle_input(uid, msg.get("dy", 0.0))


def init_agent_with_random_weights():
    """Initialize the AI agent and pipe controllers with random weights"""
    global agent, env, pipe_controllers, current_checkpoint_path

    device = get_device()
    env = FlappyEnv(
        render=False,  # Headless mode - we render in browser instead
        seed=1,
        step_penalty=-0.01,
        mute=True,
        include_gap_vel=True,
    )

    # DON'T change window.width - it affects AI state normalization!
    # Keep everything at 288px for AI, only modify pipe spawning/removal for wider view
    original_width = env.config.window.width  # 288px
    print(
        f"✓ Window width kept at {original_width}px for correct AI state calculations"
    )

    # Modify pipe spawning to keep spawning pipes even when they're beyond 288px
    # This fills the wider browser view without affecting AI state calculations
    env.pipes.can_spawn_pipes = lambda: can_spawn_pipes_modified_impl(env)

    # Also modify remove_old_pipes to keep pipes visible in wider view
    env.pipes.remove_old_pipes = lambda: remove_old_pipes_modified_impl(
        env, users, user_queue
    ) or _assign_waiting_users_to_pipes(env)

    # NOTE: Do not override make_random_pipes here; keep core env logic unchanged
    # We will fill additional visual-only pipes after each env.reset() in ai_game_loop()

    # Initialize bird DQN agent with random weights
    state_dim = 8  # 6 base features + 2 gap velocities
    action_dim = 2
    cfg = DQNConfig(state_dim=state_dim, action_dim=action_dim, device=device)
    agent = DQNAgent(cfg)
    # Don't load checkpoint - use random weights
    current_checkpoint_path = None
    print(f"✓ AI agent initialized with random weights (no checkpoint loaded)")

    # Initialize pipe controllers
    pipe_cfg = PipeControllerConfig(
        state_dim=5,  # [dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y]
        device=device,
        num_controllers=4,
    )
    pipe_controllers = PipeControllerAgent(pipe_cfg)
    print("✓ Pipe controllers initialized (no checkpoint found, starting fresh)")


def init_agent(checkpoint_path: str):
    """Initialize the AI agent and pipe controllers"""
    global agent, env, pipe_controllers, current_checkpoint_path

    device = get_device()
    env = FlappyEnv(
        render=False,  # Headless mode - we render in browser instead
        seed=1,
        step_penalty=-0.01,
        mute=True,
        include_gap_vel=True,
    )

    # DON'T change window.width - it affects AI state normalization!
    # Keep everything at 288px for AI, only modify pipe spawning/removal for wider view
    original_width = env.config.window.width  # 288px
    print(
        f"✓ Window width kept at {original_width}px for correct AI state calculations"
    )

    # Modify pipe spawning to keep spawning pipes even when they're beyond 288px
    # This fills the wider browser view without affecting AI state calculations
    env.pipes.can_spawn_pipes = lambda: can_spawn_pipes_modified_impl(env)

    # Also modify remove_old_pipes to keep pipes visible in wider view
    env.pipes.remove_old_pipes = lambda: remove_old_pipes_modified_impl(
        env, users, user_queue
    ) or _assign_waiting_users_to_pipes(env)

    # NOTE: Do not override make_random_pipes here; keep core env logic unchanged
    # We will fill additional visual-only pipes after each env.reset() in ai_game_loop()

    # Initialize bird DQN agent
    state_dim = 8  # 6 base features + 2 gap velocities
    action_dim = 2
    cfg = DQNConfig(state_dim=state_dim, action_dim=action_dim, device=device)
    agent = DQNAgent(cfg)
    agent.load(checkpoint_path)
    current_checkpoint_path = checkpoint_path  # Track loaded checkpoint
    print(f"✓ AI agent loaded from {checkpoint_path}")

    # Initialize pipe controllers
    pipe_cfg = PipeControllerConfig(
        state_dim=5,  # [dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y]
        device=device,
        num_controllers=4,
    )
    pipe_controllers = PipeControllerAgent(pipe_cfg)
    
    # Try to load pipe controller checkpoints if they exist
    # Check both locations: checkpoints/pipe_controllers (from co-training) 
    # and checkpoints/pipe_controllers (from web eval)
    pipe_checkpoint_dir = Path("checkpoints/pipe_controllers")
    if pipe_checkpoint_dir.exists():
        try:
            pipe_controllers.load(str(pipe_checkpoint_dir))
            # Clear the replay buffer for web evaluation (we want to fill it from deaths)
            pipe_controllers.replay_buffer["pos"] = 0
            pipe_controllers.replay_buffer["full"] = False
            print(f"✓ Pipe controllers loaded from {pipe_checkpoint_dir}")
            print(f"✓ Replay buffer cleared (will fill from game deaths)")
        except Exception as e:
            print(f"⚠ Could not load pipe controllers: {e}, starting fresh")
    else:
        print("✓ Pipe controllers initialized (no checkpoint found, starting fresh)")
        print("  Tip: Run 'python -m rl.train_adversarial' to pretrain both networks together")


async def broadcast(data: dict):
    """Send data to all connected clients"""
    dead_clients = set()
    for ws in clients.copy():
        try:
            await ws.send_json(data)
        except Exception:
            dead_clients.add(ws)

    clients.difference_update(dead_clients)


def extract_game_state() -> dict:
    """Extract current game state from the environment"""
    if env is None:
        # Return empty but valid structure when env is not initialized
        return {
            "bird": {"x": 0.0, "y": 0.0, "rotation": 0.0, "vel_y": 0.0},
            "pipes": [],
            "score": 0,
            "floor_x": 0.0,
            "width": 288,
            "height": 512,
            "viewport_height": 512,
        }

    # Get bird state
    bird_data = {
        "x": float(env.player.x),
        "y": float(env.player.y),
        "rotation": float(env.player.rot),
        "vel_y": float(env.player.vel_y),
    }

    # Get pipes
    pipes_data = []
    for up_pipe, low_pipe in zip(env.pipes.upper, env.pipes.lower):
        pipes_data.append(
            {
                "id": int(getattr(up_pipe, "pair_id", -1)),
                "x": float(up_pipe.x),
                "upper_y": float(up_pipe.y),
                "lower_y": float(low_pipe.y),
                "gap": float(env.pipes.pipe_gap),
                "owner": getattr(up_pipe, "ownerUserId", None),
            }
        )

    # Get floor position
    floor_x = float(env.floor.x) if hasattr(env.floor, "x") else 0

    return {
        "bird": bird_data,
        "pipes": pipes_data,
        "score": env.score.score,
        "floor_x": floor_x,
        "width": env.window.width,
        "height": env.window.height,
        "viewport_height": env.window.viewport_height,
    }


# Fill far-right with extra pipes so the browser can render many pipes at once
# without changing AI state semantics. Uses the environment's make_random_pipes
# and simply places additional pairs at increasing x positions.
def _fill_wide_view_pipes(
    env, target_last_x: int = 2000, spacing: int = 180
) -> None:
    # Ensure there is at least one pipe pair to base positions on
    if not env.pipes.upper:
        up, low = env.pipes.make_random_pipes()
        env.pipes.upper.append(up)
        env.pipes.lower.append(low)

    # Ensure each pair has metadata: pair_id and ownerUserId
    _ensure_pair_metadata(env)

    last_x = env.pipes.upper[-1].x
    while last_x + spacing < target_last_x:
        up, low = env.pipes.make_random_pipes()
        last_x += spacing
        up.x = last_x
        low.x = last_x
        # metadata
        _attach_pair_metadata(up, low)
        env.pipes.upper.append(up)
        env.pipes.lower.append(low)


def _attach_pair_metadata(up, low) -> None:
    global next_pair_id
    pair_id = getattr(up, "pair_id", None)
    if pair_id is None:
        pair_id = next_pair_id
        next_pair_id += 1
    setattr(up, "pair_id", pair_id)
    setattr(low, "pair_id", pair_id)
    # mark no owner initially
    if not hasattr(up, "ownerUserId"):
        setattr(up, "ownerUserId", None)
    if not hasattr(low, "ownerUserId"):
        setattr(low, "ownerUserId", None)


def _ensure_pair_metadata(env) -> None:
    for up, low in zip(env.pipes.upper, env.pipes.lower):
        _attach_pair_metadata(up, low)


def _assign_waiting_users_to_pipes(env) -> None:
    # Find unowned pipes (by pair) and assign to queued users
    if not user_queue:
        return
    # Visible right edge in game coordinates (canvas width 1728 / SCALE 2 = 864)
    visible_right = 864
    pairs = []
    for up, low in zip(env.pipes.upper, env.pipes.lower):
        owner = getattr(up, "ownerUserId", None)
        pairs.append(
            {
                "x": float(up.x),
                "up": up,
                "low": low,
                "owned": owner is not None,
            }
        )
    # Prefer the next unowned pipe that will enter the screen from the right:
    # smallest x >= visible_right
    future_candidates = [
        p for p in pairs if (not p["owned"]) and p["x"] >= visible_right
    ]
    future_candidates.sort(key=lambda p: p["x"])  # soonest to appear first
    # Fallback: far-right unowned pipes if none beyond threshold
    if not future_candidates:
        future_candidates = [p for p in pairs if not p["owned"]]
        future_candidates.sort(
            key=lambda p: p["x"]
        )  # still assign left-to-right
    while user_queue and future_candidates:
        p = future_candidates.pop(0)
        uid = user_queue.pop(0)
        setattr(p["up"], "ownerUserId", uid)
        setattr(p["low"], "ownerUserId", uid)
        users.setdefault(
            uid, {"joined": True, "assigned": None, "last_dy": 0.0}
        )
        users[uid]["assigned"] = int(getattr(p["up"], "pair_id", -1))


async def ai_game_loop():
    """Main AI game loop that broadcasts state and collects experiences"""
    global env, agent, training_step_counter, current_game_pipe_experiences

    if env is None or agent is None:
        print("Error: Environment or agent not initialized")
        return

    s = env.reset()
    prev_s = np.copy(s) if isinstance(s, np.ndarray) else s
    prev_reward = 0.0
    prev_pipe_states = {}  # Store pipe states/actions from previous step

    # After reset, add extra pipes to the far right for wide browser rendering
    _fill_wide_view_pipes(env, target_last_x=2000, spacing=180)

    # Bird position is correct after reset (no need to fix)

    while True:
        # Pick action & step
        a = _select_action(agent, s)
        s2, r, done, info = env.step(a)

        # Post-step maintenance and assignments (this applies pipe controls)
        _after_step_maintenance(env)

        # Collect experiences for bird DQN
        bird_experiences.append({
            "state": prev_s,
            "action": a,
            "reward": r,
            "next_state": s2,
            "done": done,
        })

        # Collect experiences for pipe controllers from PREVIOUS step
        # (pipes act AFTER bird step, so their action affects NEXT step's reward)
        # Store in current_game_pipe_experiences - will be pushed to buffer on death
        if not prev_pipe_states:
            # Debug: log if no pipe states (pipes might not be controlled)
            if training_step_counter % 100 == 0:  # Log every 100 steps to avoid spam
                nn_pipes = getattr(apply_user_inputs_to_pipes, "_last_nn_pipes", {})
                if not nn_pipes:
                    print(f"⚠ No NN-controlled pipes detected (step {training_step_counter})")
        
        for pipe_idx, pipe_data in prev_pipe_states.items():
            if pipe_idx < len(env.pipes.upper):
                # Get current state (next_state for the pipe action)
                next_state = _extract_pipe_controller_state(env, pipe_idx)
                if next_state is not None:
                    # ADVERSARIAL REWARD: Only for outcomes pipes can control
                    # Pipes can control: bird passing pipe, bird hitting pipe
                    # Pipes CANNOT control: bird flapping, time passing
                    # 
                    # Extract only pipe-influenced rewards:
                    # - Bird passes pipe (+1.0) → Pipe gets -1.0 (bad, avoid)
                    # - Bird dies (-1.0) → Pipe gets +1.0 (good, maximize!)
                    # - Step penalty (-0.01) → Pipe gets 0.0 (can't control time)
                    # - Flap cost (-0.003) → Pipe gets 0.0 (can't control bird's actions)
                    pipe_reward = 0.0
                    if r > 0.5:  # Bird passed a pipe
                        pipe_reward = -1.0
                    elif r < -0.5:  # Bird died
                        pipe_reward = 1.0
                    # Otherwise: step penalty or flap cost - pipes can't control these, so 0.0
                    
                    # No position reward - pipes only care about killing the bird
                    adversarial_reward = pipe_reward
                    current_game_pipe_experiences.append({
                        "controller_idx": pipe_idx,
                        "state": pipe_data["state"],
                        "action": pipe_data["action"],
                        "reward": adversarial_reward,  # Explicitly adversarial: higher = better for pipes
                        "next_state": next_state,
                        "done": done,
                    })

        # Store current pipe states/actions for NEXT step's reward assignment
        nn_pipes = getattr(apply_user_inputs_to_pipes, "_last_nn_pipes", {})
        prev_pipe_states = nn_pipes.copy()

        prev_s = np.copy(s2) if isinstance(s2, np.ndarray) else s2
        s = s2
        prev_reward = r
        training_step_counter += 1

        # Broadcast
        state = extract_game_state()
        current_score = env.score.score
        prev_score = getattr(ai_game_loop, "prev_score", 0)
        scored = current_score > prev_score
        ai_game_loop.prev_score = current_score
        # Get checkpoint name for display
        checkpoint_name = None
        if current_checkpoint_path:
            checkpoint_name = Path(current_checkpoint_path).name
        
        await broadcast(broadcast_frame_state(a, scored, state, checkpoint_name))

        # Push pipe experiences when bird passes a pipe (scored) OR dies (done)
        # This ensures pipes see both success (+1.0) and failure (-1.0) cases
        if scored or done:
            # Push pipe experiences from recent steps (last few steps before score/death)
            # Only push experiences that have non-zero reward (meaningful outcomes)
            if pipe_controllers is not None and current_game_pipe_experiences:
                # Filter to only push experiences with non-zero reward (pass or death)
                meaningful_experiences = [
                    exp for exp in current_game_pipe_experiences
                    if abs(exp["reward"]) > 0.01  # Only +1.0 (death) or -1.0 (pass)
                ]
                
                if meaningful_experiences:
                    buffer_len_before = pipe_controllers._buffer_len(pipe_controllers.replay_buffer)
                    for exp in meaningful_experiences:
                        pipe_controllers.push_experience(
                            exp["controller_idx"],
                            exp["state"],
                            exp["action"],
                            exp["reward"],
                            exp["next_state"],
                            exp["done"],
                        )
                    buffer_len_after = pipe_controllers._buffer_len(pipe_controllers.replay_buffer)
                    event_type = "death" if done else "pipe pass"
                    print(f"✓ Pushed {len(meaningful_experiences)} pipe experiences to buffer on {event_type} (buffer: {buffer_len_before} → {buffer_len_after})")
                    
                    # Auto-train if buffer is ready (if enabled) - ONLY on death
                    # This allows the game to pause, train, load checkpoint, and restart cleanly
                    if done and os.getenv("ENABLE_AUTO_TRAINING", "false").lower() == "true":
                        auto_train_threshold = pipe_controllers.cfg.batch_size * 2  # 2x threshold
                        if buffer_len_after >= auto_train_threshold:
                            # Run training asynchronously (don't block game loop)
                            # Small delay to let gauge update first (gauge polls every 2s)
                            await asyncio.sleep(0.5)
                            asyncio.create_task(_auto_train_on_death())
                
                # Clear experiences after pushing (but keep collecting for next event)
                # Only clear all on death, keep recent ones for next pipe pass
                if done:
                    current_game_pipe_experiences.clear()
                else:
                    # On pipe pass, keep only the last few experiences (for next pass/death)
                    # Keep last 50 experiences to maintain some history
                    if len(current_game_pipe_experiences) > 50:
                        current_game_pipe_experiences[:] = current_game_pipe_experiences[-50:]

        if done:
            
            await broadcast(
                {
                    "type": "game_over",
                    "score": info.get("score", 0),
                    "events": {"hit": True, "die": True},
                }
            )
            await asyncio.sleep(1)
            s = env.reset()
            prev_s = np.copy(s) if isinstance(s, np.ndarray) else s
            _on_game_over(env)
            ai_game_loop.prev_score = 0
            # Clear pipe experiences for new game
            current_game_pipe_experiences.clear()

        await asyncio.sleep(1 / 30)  # 30 FPS


async def training_loop():
    """Background training loop that trains both bird and pipe networks"""
    global agent, pipe_controllers, training_step_counter
    
    TRAIN_EVERY_STEPS = 100  # Train every 100 game steps
    BIRD_BATCH_SIZE = 64
    PIPE_BATCH_SIZE = 64
    CHECKPOINT_EVERY_STEPS = 10000  # Save checkpoints every 10k steps
    
    last_checkpoint_step = 0
    
    while True:
        await asyncio.sleep(1.0)  # Check every second
        
        if agent is None or pipe_controllers is None:
            continue
        
        # Train bird DQN
        if len(bird_experiences) >= BIRD_BATCH_SIZE:
            # Sample batch from recent experiences
            batch_indices = np.random.choice(
                len(bird_experiences), 
                size=min(BIRD_BATCH_SIZE, len(bird_experiences)), 
                replace=False
            )
            batch = {
                "state": np.array([bird_experiences[i]["state"] for i in batch_indices]),
                "action": np.array([bird_experiences[i]["action"] for i in batch_indices]).reshape(-1, 1),
                "reward": np.array([bird_experiences[i]["reward"] for i in batch_indices]).reshape(-1, 1),
                "next_state": np.array([bird_experiences[i]["next_state"] for i in batch_indices]),
                "done": np.array([bird_experiences[i]["done"] for i in batch_indices]).reshape(-1, 1),
            }
            try:
                loss = agent.optimize(batch)
                if training_step_counter % TRAIN_EVERY_STEPS == 0 and loss is not None:
                    print(f"Bird DQN training: step={training_step_counter}, loss={loss:.4f}")
            except Exception as e:
                print(f"Error training bird DQN: {e}")
        
        # Train pipe controllers
        # Note: Pipe experiences are now pushed to buffer on death (in ai_game_loop)
        # This training loop just trains from the buffer
        
        # Train if we have enough experiences in replay buffer
        try:
            loss = pipe_controllers.optimize()
            if training_step_counter % TRAIN_EVERY_STEPS == 0 and loss is not None:
                print(f"Pipe controllers training: step={training_step_counter}, loss={loss:.4f}")
        except Exception as e:
            if "Not enough samples" not in str(e):
                print(f"Error training pipe controllers: {e}")
        
        # Save checkpoints periodically
        if training_step_counter - last_checkpoint_step >= CHECKPOINT_EVERY_STEPS:
            try:
                # Save bird agent
                bird_ckpt_path = Path("checkpoints/best_online.pt")
                agent.save(str(bird_ckpt_path))
                
                # Save pipe controllers
                pipe_ckpt_dir = Path("checkpoints/pipe_controllers")
                pipe_ckpt_dir.mkdir(parents=True, exist_ok=True)
                pipe_controllers.save(str(pipe_ckpt_dir))
                
                print(f"✓ Checkpoints saved at step {training_step_counter}")
                last_checkpoint_step = training_step_counter
            except Exception as e:
                print(f"Error saving checkpoints: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize on server startup"""
    try:
        print("="*60, flush=True)
        print("SERVER STARTUP", flush=True)
        print("="*60, flush=True)
        sys.stdout.flush()
        checkpoint = Path("checkpoints/best.pt")
        if checkpoint.exists():
            print(f"Found checkpoint: {checkpoint}", flush=True)
            sys.stdout.flush()
            init_agent(str(checkpoint))
        else:
            print(f"Warning: Checkpoint not found at {checkpoint}", flush=True)
            sys.stdout.flush()
            # Initialize with random weights so game can still run
            print("Initializing agent with random weights...", flush=True)
            sys.stdout.flush()
            init_agent_with_random_weights()
        
        # Start game loop (always)
        print("Starting game loop...", flush=True)
        sys.stdout.flush()
        asyncio.create_task(ai_game_loop())
        
        # Start training loop only if enabled (disabled by default to avoid conflicts with main training)
        enable_web_training = os.getenv("ENABLE_WEB_TRAINING", "false").lower() == "true"
        if enable_web_training:
            print("Starting web evaluation training loop...", flush=True)
            sys.stdout.flush()
            asyncio.create_task(training_loop())
            print("⚠️  WARNING: Web eval training is enabled - this may conflict with main training!", flush=True)
        else:
            print("Web evaluation training is DISABLED (set ENABLE_WEB_TRAINING=true to enable)", flush=True)
            print("  This prevents conflicts with the main training script", flush=True)
        
        print("✓ Server startup complete - game loop running", flush=True)
        print("="*60, flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR in startup_event: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()


@app.get("/")
async def get_client():
    """Serve the browser client"""
    html_file = Path(__file__).parent / "static" / "index.html"
    html_content = html_file.read_text()
    return HTMLResponse(content=html_content)


@app.get("/api/checkpoints/bird")
async def list_bird_checkpoints():
    """List all available bird checkpoints"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return JSONResponse({"checkpoints": [], "current": None})
    
    # Find all bird checkpoint files
    bird_ckpts = []
    for pattern in ["bird_ckpt_step_*.pt", "best*.pt", "final*.pt"]:
        bird_ckpts.extend(checkpoints_dir.glob(pattern))
    
    # Sort by modification time (newest first)
    bird_ckpts = sorted(bird_ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
    
    checkpoints = [{"name": ckpt.name, "path": str(ckpt)} for ckpt in bird_ckpts]
    current = Path(current_checkpoint_path).name if current_checkpoint_path else None
    
    return JSONResponse({"checkpoints": checkpoints, "current": current})


@app.get("/api/checkpoints/pipes")
async def list_pipe_checkpoints():
    """List pipe controller checkpoint info"""
    pipe_dir = Path("checkpoints/pipe_controllers")
    if not pipe_dir.exists():
        return JSONResponse({"available": False, "current": None})
    
    pipe_files = list(pipe_dir.glob("pipe_controller_*.pt"))
    if not pipe_files:
        return JSONResponse({"available": False, "current": None})
    
    # Get optimize_steps from first controller
    try:
        ckpt = torch.load(pipe_files[0], map_location="cpu")
        optimize_steps = ckpt.get("optimize_steps", 0)
        return JSONResponse({
            "available": True,
            "current": f"{len(pipe_files)} controllers ({optimize_steps} steps)"
        })
    except:
        return JSONResponse({
            "available": True,
            "current": f"{len(pipe_files)} controllers"
        })


@app.post("/api/checkpoints/bird/load")
async def load_bird_checkpoint(request: Dict[str, Any]):
    """Load a bird checkpoint"""
    checkpoint_path = request.get("path")
    if not checkpoint_path:
        raise HTTPException(status_code=400, detail="Missing checkpoint path")
    
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    try:
        global agent, current_checkpoint_path
        device = get_device()
        state_dim = 8
        action_dim = 2
        cfg = DQNConfig(state_dim=state_dim, action_dim=action_dim, device=device)
        agent = DQNAgent(cfg)
        agent.load(str(checkpoint_file))
        current_checkpoint_path = str(checkpoint_file)
        
        # Reload pipe controllers if they exist (keep them in sync)
        pipe_checkpoint_dir = Path("checkpoints/pipe_controllers")
        if pipe_checkpoint_dir.exists():
            # Initialize pipe controllers if they don't exist
            global pipe_controllers
            if pipe_controllers is None:
                pipe_cfg = PipeControllerConfig(
                    state_dim=5,
                    device=device,
                    num_controllers=4,
                )
                pipe_controllers = PipeControllerAgent(pipe_cfg)
                print(f"✓ Initialized pipe controllers (were None)")
            
            try:
                pipe_controllers.load(str(pipe_checkpoint_dir))
            except Exception as e:
                print(f"Warning: Could not reload pipe controllers: {e}")
        
        return JSONResponse({"success": True, "checkpoint": checkpoint_file.name})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading checkpoint: {str(e)}")


@app.get("/api/training/status")
async def get_training_status():
    """Get pipe buffer status for training readiness"""
    global pipe_controllers, auto_training_active
    
    min_batch_size = 64
    auto_train_threshold = min_batch_size * 2  # 2x threshold for auto-training
    
    if pipe_controllers is None:
        return JSONResponse({
            "ready": False,
            "buffer_size": 0,
            "min_batch_size": min_batch_size,
            "auto_train_threshold": auto_train_threshold,
            "progress": 0.0,
            "auto_training": False
        })
    
    try:
        buffer_len = pipe_controllers._buffer_len(pipe_controllers.replay_buffer)
        
        # Progress is based on reaching auto-train threshold (2x)
        progress = min(1.0, buffer_len / auto_train_threshold)
        ready = buffer_len >= min_batch_size  # Manual training ready at 64
        auto_ready = buffer_len >= auto_train_threshold  # Auto-training ready at 128
        
        return JSONResponse({
            "ready": ready,
            "auto_ready": auto_ready,
            "buffer_size": buffer_len,
            "min_batch_size": min_batch_size,
            "auto_train_threshold": auto_train_threshold,
            "progress": progress,
            "auto_training": auto_training_active
        })
    except Exception as e:
        return JSONResponse({
            "ready": False,
            "buffer_size": 0,
            "min_batch_size": min_batch_size,
            "progress": 0.0,
            "error": str(e)
        })


# Global flag for auto-training status
auto_training_active = False
auto_training_rounds_completed = 0  # Track total auto-training rounds completed

async def _auto_train_on_death():
    """Automatically run training when buffer is ready after death"""
    global agent, pipe_controllers, bird_experiences, auto_training_active
    
    if agent is None or pipe_controllers is None:
        return
    
    # Small delay to avoid blocking
    await asyncio.sleep(0.1)
    
    try:
        buffer_len = pipe_controllers._buffer_len(pipe_controllers.replay_buffer)
        auto_train_threshold = pipe_controllers.cfg.batch_size * 2  # 2x threshold
        if buffer_len < auto_train_threshold:
            return
        
        # Check if bird buffer has enough experiences
        if len(bird_experiences) < 64:
            return
        
        # Set flag to indicate training is active
        auto_training_active = True
        
        # Broadcast training start to all clients
        await broadcast({
            "type": "auto_training",
            "status": "started",
            "buffer_size": buffer_len
        })
        
        print(f"🤖 Auto-training triggered (buffer: {buffer_len}, bird buffer: {len(bird_experiences)})")
        
        # Run 5 rounds of training (bird + pipes)
        NUM_ROUNDS = 5
        BIRD_BATCH_SIZE = 64
        
        for round_num in range(1, NUM_ROUNDS + 1):
            # Train bird
            if len(bird_experiences) >= BIRD_BATCH_SIZE:
                batch_indices = np.random.choice(
                    len(bird_experiences),
                    size=min(BIRD_BATCH_SIZE, len(bird_experiences)),
                    replace=False
                )
                batch = {
                    "state": np.array([bird_experiences[i]["state"] for i in batch_indices]),
                    "action": np.array([bird_experiences[i]["action"] for i in batch_indices]).reshape(-1, 1),
                    "reward": np.array([bird_experiences[i]["reward"] for i in batch_indices]).reshape(-1, 1),
                    "next_state": np.array([bird_experiences[i]["next_state"] for i in batch_indices]),
                    "done": np.array([bird_experiences[i]["done"] for i in batch_indices]).reshape(-1, 1),
                }
                try:
                    loss = agent.optimize(batch)
                    if loss is not None:
                        print(f"  Round {round_num}/5 - Bird loss: {loss:.4f}")
                except Exception as e:
                    print(f"  Bird training error (round {round_num}): {e}")
            
            # Train pipes
            try:
                loss = pipe_controllers.optimize()
                if loss is not None:
                    print(f"  Round {round_num}/5 - Pipe loss: {loss:.4f}")
            except Exception as e:
                if "Not enough samples" not in str(e):
                    print(f"  Pipe training error (round {round_num}): {e}")
            
            # Yield control back to event loop between rounds to prevent blocking
            if round_num < NUM_ROUNDS:
                await asyncio.sleep(0.01)  # Small delay to allow game loop to continue
        
        print(f"✓ Completed {NUM_ROUNDS} training rounds")
        
        # Clear buffer after training (so gauge resets and tracks new experiences)
        pipe_controllers.replay_buffer["pos"] = 0
        pipe_controllers.replay_buffer["full"] = False
        
        # Save checkpoints after training
        try:
            # Save bird agent to best_online.pt
            bird_ckpt_path = Path("checkpoints/best_online.pt")
            bird_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(bird_ckpt_path))
            
            # Save pipe controllers
            pipe_ckpt_dir = Path("checkpoints/pipe_controllers")
            pipe_ckpt_dir.mkdir(parents=True, exist_ok=True)
            pipe_controllers.save(str(pipe_ckpt_dir))
            
            # Update current checkpoint path
            global current_checkpoint_path
            current_checkpoint_path = str(bird_ckpt_path)
            
            print(f"✓ Saved checkpoints: best_online.pt and pipe_controllers")
        except Exception as e:
            print(f"⚠ Error saving checkpoints after auto-training: {e}")
        
        # Clear flag and broadcast training end
        auto_training_active = False
        global auto_training_rounds_completed
        auto_training_rounds_completed += 1
        await broadcast({
            "type": "auto_training",
            "status": "completed",
            "rounds_completed": auto_training_rounds_completed
        })
        
        print(f"✓ Auto-training complete (total rounds: {auto_training_rounds_completed}, buffer cleared, ready for new experiences)")
    except Exception as e:
        auto_training_active = False
        await broadcast({
            "type": "auto_training",
            "status": "error",
            "error": str(e)
        })
        print(f"Error in auto-training: {e}")


def _run_pipe_training_episodes(num_episodes: int = 5) -> list:
    """
    Run episodes with greedy bird and collect pipe experiences for training.
    Returns list of pipe experiences from these episodes.
    """
    global env, agent, pipe_controllers
    
    if env is None or agent is None or pipe_controllers is None:
        return []
    
    pipe_training_experiences = []
    
    for episode in range(num_episodes):
        s = env.reset()
        prev_pipe_states = {}
        done = False
        
        while not done:
            # Bird uses greedy policy (no exploration)
            a = _select_action(agent, s)  # Greedy action
            
            # Apply pipe controls (pipes act)
            _after_step_maintenance(env)
            
            # Step environment
            s2, r, done, info = env.step(a)
            
            # Collect pipe experiences from PREVIOUS step
            for pipe_idx, pipe_data in prev_pipe_states.items():
                if pipe_idx < len(env.pipes.upper):
                    next_state = _extract_pipe_controller_state(env, pipe_idx)
                    if next_state is not None:
                        # Adversarial reward
                        pipe_reward = 0.0
                        if r > 0.5:  # Bird passed a pipe
                            pipe_reward = -1.0
                        elif r < -0.5:  # Bird died
                            pipe_reward = 1.0
                        
                        pipe_training_experiences.append({
                            "controller_idx": pipe_idx,
                            "state": pipe_data["state"],
                            "action": pipe_data["action"],
                            "reward": pipe_reward,
                            "next_state": next_state,
                            "done": done,
                        })
            
            # Store current pipe states/actions for NEXT step
            nn_pipes = getattr(apply_user_inputs_to_pipes, "_last_nn_pipes", {})
            prev_pipe_states = nn_pipes.copy()
            
            s = s2
    
    return pipe_training_experiences


@app.post("/api/training/run-rounds")
async def run_training_rounds(request: Dict[str, Any]):
    """Run a few rounds of alternative training (bird then pipe)"""
    global agent, pipe_controllers, bird_experiences
    
    num_rounds = request.get("rounds", 3)  # Default 3 rounds
    rounds_per_agent = request.get("rounds_per_agent", 2)  # Default 2 training steps per agent per round
    
    if agent is None or pipe_controllers is None:
        raise HTTPException(status_code=400, detail="Agent or pipe controllers not initialized")
    
    # Check if pipe buffer has enough experiences (from deaths)
    buffer_len = pipe_controllers._buffer_len(pipe_controllers.replay_buffer)
    if buffer_len < pipe_controllers.cfg.batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Pipe buffer not ready: {buffer_len}/{pipe_controllers.cfg.batch_size} (need more deaths)"
        )
    
    results = {
        "rounds_completed": 0,
        "bird_losses": [],
        "pipe_losses": [],
        "errors": []
    }
    
    BIRD_BATCH_SIZE = 64
    
    try:
        for round_num in range(num_rounds):
            # Round 1: Train Bird (uses experiences from buffer)
            for _ in range(rounds_per_agent):
                if len(bird_experiences) >= BIRD_BATCH_SIZE:
                    batch_indices = np.random.choice(
                        len(bird_experiences),
                        size=min(BIRD_BATCH_SIZE, len(bird_experiences)),
                        replace=False
                    )
                    batch = {
                        "state": np.array([bird_experiences[i]["state"] for i in batch_indices]),
                        "action": np.array([bird_experiences[i]["action"] for i in batch_indices]).reshape(-1, 1),
                        "reward": np.array([bird_experiences[i]["reward"] for i in batch_indices]).reshape(-1, 1),
                        "next_state": np.array([bird_experiences[i]["next_state"] for i in batch_indices]),
                        "done": np.array([bird_experiences[i]["done"] for i in batch_indices]).reshape(-1, 1),
                    }
                    try:
                        loss = agent.optimize(batch)
                        if loss is not None:
                            results["bird_losses"].append(float(loss))
                    except Exception as e:
                        results["errors"].append(f"Bird training error: {str(e)}")
            
            # Round 2: Train Pipes (using experiences from buffer - games where pipes killed the bird)
            # Train pipes on experiences collected from deaths
            for _ in range(rounds_per_agent):
                try:
                    loss = pipe_controllers.optimize()
                    if loss is not None:
                        results["pipe_losses"].append(float(loss))
                except Exception as e:
                    if "Not enough samples" not in str(e):
                        results["errors"].append(f"Pipe training error: {str(e)}")
            
            results["rounds_completed"] += 1
        
        # Clear buffer after training (so gauge resets and tracks new experiences)
        pipe_controllers.replay_buffer["pos"] = 0
        pipe_controllers.replay_buffer["full"] = False
        
        return JSONResponse({
            "success": True,
            "results": results,
            "message": "Training complete - buffer cleared, ready for new experiences"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming game state"""
    await websocket.accept()
    clients.add(websocket)

    try:
        # Send initial state
        _ws_send_init(websocket)

        # Keep connection alive and process client messages
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            process_ws_message(msg, websocket)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
