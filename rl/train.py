import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.ai_env import FlappyEnv
from .dqn_agent import DQNAgent, DQNConfig
from .replay_buffer import ReplayBuffer


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args: argparse.Namespace) -> None:
    device = get_device()

    env = FlappyEnv(
        render=args.render,
        seed=args.seed,
        step_penalty=args.step_penalty,
        mute=args.mute,
        flap_cost=args.flap_cost,
        out_of_bounds_cost=args.out_of_bounds_cost,
        moving_gaps=args.moving_gaps,
        gap_amp_px=args.gap_amp_px,
        gap_freq_hz=args.gap_freq_hz,
        include_gap_vel=args.include_gap_vel,
        center_reward=args.center_reward,
    )
    state_dim = 6 + (2 if args.include_gap_vel else 0)
    action_dim = 2

    cfg = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        target_update_every=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay,
        grad_clip=args.grad_clip,
        device=device,
    )
    agent = DQNAgent(cfg)

    if args.resume and Path(args.resume).exists():
        agent.load(args.resume)

    buf = ReplayBuffer(state_dim=state_dim, capacity=args.capacity, seed=args.seed or 0)

    checkpoints = Path(args.checkpoints)
    checkpoints.mkdir(parents=True, exist_ok=True)

    s = env.reset()
    episode = 1
    ep_reward = 0.0
    best_ma = -1e9
    scores = []
    t0 = time.time()

    for step in range(1, args.train_steps + 1):
        if step < args.warmup_steps:
            a = int(np.random.randint(0, action_dim))
        else:
            a = agent.epsilon_greedy(s)

        # Frame skip: repeat the same action for N steps, accumulate reward
        total_r = 0.0
        for _ in range(max(1, args.frameskip)):
            s2, r, done, info = env.step(a)
            info["epsilon"] = agent.eps
            if args.render:
                env.draw_hud(info)
            total_r += r
            if done:
                break

        buf.push(s, a, total_r, s2, done)
        s = s2
        ep_reward += total_r

        if step >= args.warmup_steps and len(buf) >= args.batch_size and step % args.optimize_every == 0:
            batch = buf.sample(args.batch_size)
            loss = agent.optimize(batch)
        else:
            loss = None

        if done:
            scores.append(info.get("score", 0))
            s = env.reset()
            episode += 1
            ep_reward = 0.0

        # Early stopping by time or episodes
        if args.max_seconds > 0 and (time.time() - t0) >= args.max_seconds:
            break
        if args.max_episodes > 0 and episode > args.max_episodes:
            break

        if step % args.log_every == 0:
            ma50 = np.mean(scores[-50:]) if scores else 0.0
            print(
                f"step={step} ep={episode} eps={agent.eps:.3f} ma50={ma50:.6f} buf={len(buf)} loss={loss if loss is not None else '-'}"
            )

            if args.viz and scores:
                try:
                    import matplotlib.pyplot as plt

                    fig = plt.figure(figsize=(6, 3))
                    ax = fig.add_subplot(111)
                    ax.plot(scores, label="score")
                    if len(scores) >= 50:
                        ma = np.convolve(scores, np.ones(50) / 50, mode="valid")
                        ax.plot(range(49, 49 + len(ma)), ma, label="ma50")
                    ax.legend()
                    ax.set_title("Training Progress")
                    fig.tight_layout()
                    out = Path(args.checkpoints) / "progress.png"
                    fig.savefig(out)
                    plt.close(fig)
                except Exception:
                    pass

        if step % args.ckpt_every == 0:
            ckpt_path = checkpoints / f"ckpt_step_{step}.pt"
            agent.save(str(ckpt_path))

            ma50 = np.mean(scores[-50:]) if scores else 0.0
            if ma50 > best_ma:
                best_ma = ma50
                agent.save(str(checkpoints / "best.pt"))

    env.close()

    # Auto-evaluate after training if requested
    if args.auto_eval:
        best_ckpt = Path(args.checkpoints) / "best.pt"
        if best_ckpt.exists():
            args.eval = str(best_ckpt)
        else:
            # try latest checkpoint by step
            ckpts = sorted(Path(args.checkpoints).glob("ckpt_step_*.pt"))
            if ckpts:
                args.eval = str(ckpts[-1])
        if args.eval:
            evaluate(args)


def evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    env = FlappyEnv(
        render=True,
        seed=args.seed,
        step_penalty=args.step_penalty,
        mute=args.mute,
        flap_cost=args.flap_cost,
        out_of_bounds_cost=args.out_of_bounds_cost,
        moving_gaps=args.moving_gaps,
        gap_amp_px=args.gap_amp_px,
        gap_freq_hz=args.gap_freq_hz,
        include_gap_vel=args.include_gap_vel,
        center_reward=args.center_reward,
    )
    # Match network input size to env state (6 base + 2 optional velocities)
    state_dim = 6 + (2 if args.include_gap_vel else 0)
    action_dim = 2
    cfg = DQNConfig(state_dim=state_dim, action_dim=action_dim, device=device)
    agent = DQNAgent(cfg)
    agent.load(args.eval)

    s = env.reset()
    while True:
        # Greedy evaluation (no exploration)
        with torch.no_grad():
            state_t = torch.from_numpy(s).float().unsqueeze(0).to(agent.device)
            q_vals = agent.q(state_t)
            a = int(torch.argmax(q_vals, dim=-1).item())
        s, _, done, info = env.step(a)
        if done:
            s = env.reset()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--mute", dest="mute", action="store_true")
    parser.add_argument("--no-mute", dest="mute", action="store_false")
    parser.set_defaults(mute=True)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--flap-cost", type=float, default=0.003, help="Penalty per flap action to reduce over-flapping")
    parser.add_argument("--out-of-bounds-cost", type=float, default=0.005, help="Penalty when the bird goes above the top of the viewport")
    parser.add_argument("--moving-gaps", action="store_true", help="Enable vertically moving pipe gaps")
    parser.add_argument("--gap-amp-px", type=float, default=20.0, help="Amplitude (pixels) of gap oscillation")
    parser.add_argument("--gap-freq-hz", type=float, default=0.5, help="Frequency (Hz) of gap oscillation")
    parser.add_argument("--include-gap-vel", action="store_true", help="Append vertical velocity of next two gaps to the state")
    parser.add_argument("--center-reward", type=float, default=0.0, help="Shaping: reward for reducing |dy1| toward gap center")

    # DQN
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=200_000)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    # Replay
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--optimize-every", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1, help="Repeat each chosen action for N frames")

    # Train
    parser.add_argument("--train-steps", type=int, default=300_000)
    parser.add_argument("--max-seconds", type=int, default=0, help="Stop after N seconds (0=disabled)")
    parser.add_argument("--max-episodes", type=int, default=0, help="Stop after N episodes (0=disabled)")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--auto-eval", action="store_true", help="After training, automatically run eval with best/latest checkpoint")

    # Eval
    parser.add_argument("--eval", type=str, default="")
    parser.add_argument("--viz", action="store_true")

    args = parser.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()


