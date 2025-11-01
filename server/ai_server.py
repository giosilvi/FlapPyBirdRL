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
from pathlib import Path
from typing import Set

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from rl.dqn_agent import DQNAgent, DQNConfig
from src.ai_env import FlappyEnv

app = FastAPI()

# Mount assets directory
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Global state
env = None
agent = None
clients: Set[WebSocket] = set()


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    def can_spawn_pipes_modified():
        last = env.pipes.upper[-1] if env.pipes.upper else None
        if not last:
            return True
        # Spawn when the last pipe has moved left enough (tighter spacing)
        # This keeps the right side of the wide view filled with pipes
        return last.x < 1300  # Keep spawning until pipes reach this x position

    env.pipes.can_spawn_pipes = can_spawn_pipes_modified

    # Also modify remove_old_pipes to keep pipes visible in wider view
    def remove_old_pipes_modified():
        # Only remove pipes that are way off-screen (past left edge)
        for pipe in list(env.pipes.upper):
            if pipe.x < -pipe.w - 200:  # Extra buffer
                env.pipes.upper.remove(pipe)
        for pipe in list(env.pipes.lower):
            if pipe.x < -pipe.w - 200:
                env.pipes.lower.remove(pipe)

    env.pipes.remove_old_pipes = remove_old_pipes_modified

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
                "x": float(up_pipe.x),
                "upper_y": float(up_pipe.y),
                "lower_y": float(low_pipe.y),
                "gap": float(env.pipes.pipe_gap),
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

    last_x = env.pipes.upper[-1].x
    while last_x + spacing < target_last_x:
        up, low = env.pipes.make_random_pipes()
        last_x += spacing
        up.x = last_x
        low.x = last_x
        env.pipes.upper.append(up)
        env.pipes.lower.append(low)


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
        # AI decision
        with torch.no_grad():
            state_t = torch.from_numpy(s).float().unsqueeze(0).to(agent.device)
            q_vals = agent.q(state_t)
            a = int(torch.argmax(q_vals, dim=-1).item())

        # Step the environment
        s, _, done, info = env.step(a)

        # Continuously keep far-right buffer of pipes so the browser always sees many
        _fill_wide_view_pipes(env, target_last_x=2100, spacing=180)

        # Broadcast game state to all connected clients
        state = extract_game_state()

        # Detect events for sound effects
        current_score = env.score.score
        prev_score = getattr(ai_game_loop, "prev_score", 0)
        scored = current_score > prev_score
        ai_game_loop.prev_score = current_score

        await broadcast(
            {
                "type": "frame",
                "state": state,
                "events": {
                    "flap": a == 1,  # Bird flapped
                    "score": scored,  # Passed a pipe
                },
            }
        )

        if done:
            await broadcast(
                {
                    "type": "game_over",
                    "score": info.get("score", 0),
                    "events": {
                        "hit": True,
                        "die": True,
                    },
                }
            )
            await asyncio.sleep(1)
            s = env.reset()
            # Refill extra far-right pipes for wide view after each reset
            _fill_wide_view_pipes(env, target_last_x=2000, spacing=180)
            # Bird position is correct after reset
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
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Flappy Bird AI - Live View</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: white;
            font-family: 'Arial', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            margin-bottom: 20px;
            color: #4ecca3;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        #container {
            position: relative;
            background: #000;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        #gameCanvas {
            border: 3px solid #4ecca3;
            display: block;
        }
        #info {
            margin-top: 15px;
            font-size: 20px;
            display: flex;
            gap: 30px;
        }
        .info-item {
            padding: 10px 20px;
            background: rgba(78, 204, 163, 0.1);
            border-radius: 5px;
            border: 1px solid #4ecca3;
        }
        .status {
            color: #4ecca3;
        }
        .error {
            color: #ff6b6b;
        }
        .label {
            opacity: 0.7;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>🐦 Flappy Bird AI - Live View</h1>
    <div id="container">
        <canvas id="gameCanvas" width="1152" height="1024"></canvas>
    </div>
    <div id="info">
        <div class="info-item">
            <div class="label">Score</div>
            <div id="score">0</div>
        </div>
        <div class="info-item">
            <div class="label">Status</div>
            <div id="status" class="status">Connecting...</div>
        </div>
        <div class="info-item">
            <div class="label">FPS</div>
            <div id="fps">0</div>
        </div>
        <div class="info-item" style="cursor: pointer;" id="muteBtn">
            <div class="label">Sound</div>
            <div id="soundStatus">🔊 On</div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const scoreEl = document.getElementById('score');
        const statusEl = document.getElementById('status');
        const fpsEl = document.getElementById('fps');
        const muteBtn = document.getElementById('muteBtn');
        const soundStatus = document.getElementById('soundStatus');

        const SCALE = 2; // Render at 2x scale for better visibility
        let gameState = null;
        let ws;
        let frameCount = 0;
        let lastFpsUpdate = Date.now();
        let birdFrame = 0;
        let birdAnimCounter = 0;
        let isMuted = false;

        // Load assets
        const images = {
            background: new Image(),
            floor: new Image(),
            pipeGreen: new Image(),
            birdMid: new Image(),
            birdUp: new Image(),
            birdDown: new Image(),
        };

        let assetsLoaded = 0;
        const totalAssets = Object.keys(images).length;

        function onImageLoad() {
            assetsLoaded++;
            if (assetsLoaded === totalAssets) {
                console.log('All assets loaded!');
            }
        }

        images.background.src = '/assets/sprites/background-day.png';
        images.background.onload = onImageLoad;
        images.floor.src = '/assets/sprites/base.png';
        images.floor.onload = onImageLoad;
        images.pipeGreen.src = '/assets/sprites/pipe-green.png';
        images.pipeGreen.onload = onImageLoad;
        images.birdMid.src = '/assets/sprites/yellowbird-midflap.png';
        images.birdMid.onload = onImageLoad;
        images.birdUp.src = '/assets/sprites/yellowbird-upflap.png';
        images.birdUp.onload = onImageLoad;
        images.birdDown.src = '/assets/sprites/yellowbird-downflap.png';
        images.birdDown.onload = onImageLoad;

        // Load audio
        const sounds = {
            wing: new Audio('/assets/audio/wing.ogg'),
            point: new Audio('/assets/audio/point.ogg'),
            hit: new Audio('/assets/audio/hit.ogg'),
            die: new Audio('/assets/audio/die.ogg'),
        };

        // Preload all sounds
        Object.values(sounds).forEach(sound => {
            sound.load();
            sound.volume = 0.3; // Lower volume to not be too loud
        });

        function playSound(soundName) {
            if (isMuted || !sounds[soundName]) return;
            try {
                sounds[soundName].currentTime = 0; // Reset to start
                sounds[soundName].play().catch(e => console.log('Audio play failed:', e));
            } catch (e) {
                console.log('Audio error:', e);
            }
        }

        // Mute button handler
        muteBtn.addEventListener('click', () => {
            isMuted = !isMuted;
            soundStatus.textContent = isMuted ? '🔇 Off' : '🔊 On';
        });

        function drawGame() {
            if (!gameState || assetsLoaded < totalAssets) return;

            const state = gameState.state;

            // Draw background (tiled)
            const bgWidth = images.background.width * SCALE;
            const bgHeight = images.background.height * SCALE;
            for (let x = 0; x < canvas.width; x += bgWidth) {
                ctx.drawImage(images.background, x, 0, bgWidth, bgHeight);
            }

            // Draw floor
            const floorY = state.viewport_height * SCALE;
            const floorWidth = images.floor.width * SCALE;
            const floorHeight = images.floor.height * SCALE;
            const floorX = (state.floor_x * SCALE) % floorWidth;
            for (let x = -floorWidth + floorX; x < canvas.width; x += floorWidth) {
                ctx.drawImage(images.floor, x, floorY, floorWidth, floorHeight);
            }

            // Draw pipes
            for (const pipe of state.pipes) {
                const x = pipe.x * SCALE;
                const pipeWidth = images.pipeGreen.width * SCALE;
                const pipeHeight = images.pipeGreen.height * SCALE;

                // Upper pipe: top-left is at (x, upper_y), rotated 180 degrees
                const upperY = pipe.upper_y * SCALE;
                ctx.save();
                ctx.translate(x + pipeWidth/2, upperY + pipeHeight/2);
                ctx.rotate(Math.PI);  // Rotate 180 degrees
                ctx.drawImage(images.pipeGreen, -pipeWidth/2, -pipeHeight/2, pipeWidth, pipeHeight);
                ctx.restore();

                // Lower pipe: top-left is at (x, lower_y), drawn normally
                const lowerY = pipe.lower_y * SCALE;
                ctx.drawImage(images.pipeGreen, x, lowerY, pipeWidth, pipeHeight);
            }

            // Draw bird with animation
            const bird = state.bird;
            const birdX = bird.x * SCALE;
            const birdY = bird.y * SCALE;

            // Animate bird flapping (cycle through frames)
            birdAnimCounter++;
            if (birdAnimCounter % 5 === 0) {
                birdFrame = (birdFrame + 1) % 3;
            }

            const birdImages = [images.birdUp, images.birdMid, images.birdDown];
            const birdImg = birdImages[birdFrame];
            const birdWidth = birdImg.width * SCALE;
            const birdHeight = birdImg.height * SCALE;

            ctx.save();
            ctx.translate(birdX + birdWidth/2, birdY + birdHeight/2);
            ctx.rotate((bird.rotation * Math.PI) / 180);
            ctx.drawImage(birdImg, -birdWidth/2, -birdHeight/2, birdWidth, birdHeight);
            ctx.restore();

            // Update FPS
            frameCount++;
            const now = Date.now();
            if (now - lastFpsUpdate > 1000) {
                fpsEl.textContent = frameCount;
                frameCount = 0;
                lastFpsUpdate = now;
            }
        }

        function animate() {
            drawGame();
            requestAnimationFrame(animate);
        }

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                statusEl.textContent = 'Connected';
                statusEl.className = 'status';
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'frame' || data.type === 'init') {
                    gameState = data;
                    if (data.state) {
                        scoreEl.textContent = data.state.score;
                    }

                    // Play sound effects based on events
                    if (data.events) {
                        if (data.events.flap) playSound('wing');
                        if (data.events.score) playSound('point');
                    }
                } else if (data.type === 'game_over') {
                    console.log('Game Over! Score:', data.score);

                    // Play death sounds
                    if (data.events) {
                        if (data.events.hit) playSound('hit');
                        if (data.events.die) {
                            setTimeout(() => playSound('die'), 100);
                        }
                    }
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                statusEl.textContent = 'Error';
                statusEl.className = 'error';
            };

            ws.onclose = () => {
                statusEl.textContent = 'Reconnecting...';
                statusEl.className = 'error';
                setTimeout(connect, 2000);
            };
        }

        connect();
        animate();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming game state"""
    await websocket.accept()
    clients.add(websocket)

    try:
        # Send initial state
        state = extract_game_state()
        await websocket.send_json({"type": "init", "state": state})

        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
