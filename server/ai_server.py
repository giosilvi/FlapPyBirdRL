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
import uuid
from pathlib import Path
from typing import Dict, Set

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from rl.dqn_agent import DQNAgent, DQNConfig
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
clients: Set[WebSocket] = set()

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


def apply_user_inputs_to_pipes(env, users: Dict[str, Dict]) -> None:
    """Apply per-user dy input to their owned pipe pair, with bounds checking."""
    _ensure_pair_metadata(env)
    pipe_gap = env.pipes.pipe_gap
    viewport_h = env.window.viewport_height
    pipe_h = env.images.pipe[0].get_height()
    for up, low in zip(env.pipes.upper, env.pipes.lower):
        owner = getattr(up, "ownerUserId", None)
        if not owner:
            continue
        u = users.get(owner)
        if not u:
            continue
        dy = float(u.get("last_dy", 0.0))
        if abs(dy) < 1e-6:
            continue
        center = low.y - pipe_gap / 2.0
        center += dy * 3.0
        min_center = pipe_gap / 2.0 + 5
        max_center = viewport_h - pipe_gap / 2.0 - 5
        if center < min_center:
            center = min_center
        if center > max_center:
            center = max_center
        up.y = center - pipe_gap / 2.0 - pipe_h
        low.y = center + pipe_gap / 2.0


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


def broadcast_frame_state(a: int, scored: bool, state: dict) -> dict:
    """Build the broadcast payload for a frame update."""
    return {
        "type": "frame",
        "state": state,
        "colors": {
            uid: u.get("color") for uid, u in users.items() if u.get("color")
        },
        "events": {"flap": a == 1, "score": scored},
    }


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


def _ws_send_init(websocket: WebSocket) -> None:
    asyncio.create_task(
        websocket.send_json(
            {
                "type": "init",
                "state": extract_game_state(),
                "colors": {
                    uid: u.get("color")
                    for uid, u in users.items()
                    if u.get("color")
                },
            }
        )
    )


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


def init_agent(checkpoint_path: str):
    """Initialize the AI agent"""
    global agent, env

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

    state_dim = 8  # 6 base + 2 gap velocities
    action_dim = 2
    cfg = DQNConfig(state_dim=state_dim, action_dim=action_dim, device=device)
    agent = DQNAgent(cfg)
    agent.load(checkpoint_path)
    print(f"✓ AI agent loaded from {checkpoint_path}")


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
        return {}

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
    """Main AI game loop that broadcasts state"""
    global env, agent

    if env is None or agent is None:
        print("Error: Environment or agent not initialized")
        return

    s = env.reset()

    # After reset, add extra pipes to the far right for wide browser rendering
    _fill_wide_view_pipes(env, target_last_x=2000, spacing=180)

    # Bird position is correct after reset (no need to fix)

    while True:
        # Pick action & step
        a = _select_action(agent, s)
        s, _, done, info = env.step(a)

        # Post-step maintenance and assignments
        _after_step_maintenance(env)

        # Broadcast
        state = extract_game_state()
        current_score = env.score.score
        prev_score = getattr(ai_game_loop, "prev_score", 0)
        scored = current_score > prev_score
        ai_game_loop.prev_score = current_score
        await broadcast(broadcast_frame_state(a, scored, state))

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
            _on_game_over(env)
            ai_game_loop.prev_score = 0

        await asyncio.sleep(1 / 30)  # 30 FPS


@app.on_event("startup")
async def startup_event():
    """Initialize on server startup"""
    checkpoint = Path("checkpoints/best.pt")
    if checkpoint.exists():
        init_agent(str(checkpoint))
        asyncio.create_task(ai_game_loop())
    else:
        print(f"Warning: Checkpoint not found at {checkpoint}")


@app.get("/")
async def get_client():
    """Serve the browser client"""
    html_file = Path(__file__).parent / "static" / "index.html"
    html_content = html_file.read_text()
    return HTMLResponse(content=html_content)


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
