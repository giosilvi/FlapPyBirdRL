"""
Co-training script for bird DQN and adversarial pipe controllers.

Trains both networks together in an adversarial setting:
- Bird DQN learns to play the game
- Pipe controllers learn to make the game harder
- Both networks train simultaneously from shared experiences
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.ai_env import FlappyEnv
from .dqn_agent import DQNAgent, DQNConfig
from .pipe_controller_agent import PipeControllerAgent, PipeControllerConfig
from .replay_buffer import ReplayBuffer


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_pipe_controller_state(env, pipe_idx: int):
    """Extract state for pipe controller: [dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y]"""
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
    
    ahead_pipe_gap_center_y_norm = 0.5
    if pipe_idx > 0 and (pipe_idx - 1) < len(env.pipes.lower):
        ahead_low = env.pipes.lower[pipe_idx - 1]
        ahead_gap_center_y = ahead_low.y - env.pipes.pipe_gap / 2.0
        ahead_pipe_gap_center_y_norm = ahead_gap_center_y / float(env.window.viewport_height)
    
    return np.array([dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y_norm], dtype=np.float32)


def apply_pipe_controls(env, pipe_controllers, pipe_greedy: bool = False):
    """Apply pipe controller actions to pipes."""
    if pipe_controllers is None:
        return {}
    
    pipe_gap = env.pipes.pipe_gap
    viewport_h = env.window.viewport_height
    pipe_h = env.images.pipe[0].get_height()
    
    nn_controlled_pipes = {}
    
    for idx, (up, low) in enumerate(zip(env.pipes.upper, env.pipes.lower)):
        if idx >= 4:  # Only control first 4 pipes
            break
        
        state = _extract_pipe_controller_state(env, idx)
        if state is not None:
            dy = pipe_controllers.act(idx, state, greedy=pipe_greedy)
            nn_controlled_pipes[idx] = {"state": state, "action": dy}
            
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
    
    return nn_controlled_pipes


def train_adversarial(args: argparse.Namespace) -> None:
    """Train both bird DQN and pipe controllers together."""
    device = get_device()

    env = FlappyEnv(
        render=args.render,
        seed=args.seed,
        step_penalty=args.step_penalty,
        mute=args.mute,
        flap_cost=args.flap_cost,
        out_of_bounds_cost=args.out_of_bounds_cost,
        moving_gaps=False,  # Pipe controllers handle movement
        include_gap_vel=True,  # Always include gap velocities (8D state)
        center_reward=args.center_reward,
    )
    
    # Initialize bird DQN
    state_dim = 8  # 6 base features + 2 gap velocities
    action_dim = 2
    
    bird_cfg = DQNConfig(
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
    bird_agent = DQNAgent(bird_cfg)
    
    if args.resume_bird and Path(args.resume_bird).exists():
        bird_agent.load(args.resume_bird)
        checkpoint_eps = bird_agent.eps
        
        # Greedy mode: set epsilon to small value (tiny exploration, mostly best policy)
        if args.greedy_mode:
            bird_agent.eps = 0.01  # Tiny exploration (1%)
            bird_agent.cfg.eps_start = 0.01
            bird_agent.cfg.eps_end = 0.01
            bird_agent.steps = 0  # Reset steps so epsilon stays at 0.01
            print(f"✓ Loaded bird agent from {args.resume_bird}")
            print(f"✓ GREEDY MODE: epsilon=0.01 (bird uses best policy with tiny exploration)")
        # Override epsilon if specified
        elif args.eps_start_override is not None:
            bird_agent.eps = args.eps_start_override
            # Reset steps so epsilon can decay from eps_start_override to eps_end
            # Calculate steps so that current epsilon matches eps_start_override
            bird_agent.steps = 0  # Reset to allow decay from override value to eps_end
            # Update cfg so decay works from override to eps_end
            bird_agent.cfg.eps_start = args.eps_start_override
            print(f"✓ Loaded bird agent from {args.resume_bird}")
            print(f"✓ Set epsilon to {args.eps_start_override}, will decay to {args.eps_end}")
        else:
            # When resuming: start from 0.05 and decay to 0.01
            resume_eps_start = 0.05
            if checkpoint_eps > resume_eps_start:
                bird_agent.eps = resume_eps_start
            else:
                bird_agent.eps = checkpoint_eps  # Use checkpoint if already lower
            
            # Reset steps counter so epsilon can decay from current value to eps_end
            bird_agent.steps = 0
            # Update cfg so decay works from resume_eps_start to eps_end
            bird_agent.cfg.eps_start = bird_agent.eps
            print(f"✓ Loaded bird agent from {args.resume_bird}")
            print(f"✓ Set epsilon to {bird_agent.eps:.3f}, will decay to {args.eps_end} during training")

    # Initialize pipe controllers
    pipe_cfg = PipeControllerConfig(
        state_dim=5,  # [dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y]
        lr=args.pipe_lr,
        batch_size=args.pipe_batch_size,
        capacity=args.pipe_capacity,
        grad_clip=args.pipe_grad_clip,
        device=device,
        num_controllers=4,
    )
    pipe_controllers = PipeControllerAgent(pipe_cfg)
    
    if args.resume_pipes and Path(args.resume_pipes).exists():
        pipe_controllers.load(args.resume_pipes)
        print(f"✓ Loaded pipe controllers from {args.resume_pipes}")

    # Replay buffers
    bird_buf = ReplayBuffer(state_dim=state_dim, capacity=args.capacity, seed=args.seed or 0)

    # Checkpoints
    checkpoints = Path(args.checkpoints)
    checkpoints.mkdir(parents=True, exist_ok=True)
    pipe_checkpoints = checkpoints / "pipe_controllers"
    pipe_checkpoints.mkdir(parents=True, exist_ok=True)

    s = env.reset()
    episode = 1
    ep_reward = 0.0
    best_ma = -1e9
    scores = []
    score_steps = []  # Track step number for each score
    prev_score = 0  # Track score to detect pipe passing
    t0 = time.time()
    
    # Track last pipe_loss across log intervals
    last_pipe_loss = None

    print(f"\n{'='*60}")
    print("Starting adversarial co-training")
    print(f"Bird DQN: {state_dim}D state, {action_dim} actions")
    print(f"Pipe Controllers: 4 networks, {pipe_cfg.state_dim}D state")
    
    # Stage-based training setup
    stage_duration = args.stage_duration
    use_stages = stage_duration > 0
    if use_stages:
        print(f"Stage-based training: {stage_duration} steps per stage")
        print("  Stage 1 (Bird Training): Bird explores & trains, Pipes greedy (no training)")
        print("  Stage 2 (Pipe Training): Bird greedy (no training), Pipes explore & train")
    else:
        print("Simultaneous training: Both agents train together")
    print(f"{'='*60}\n")

    # Stage-based epsilon settings for bird training
    bird_stage_eps_start = 0.2  # Initial epsilon during bird training stages
    bird_stage_eps_end = 0.05   # Final epsilon during bird training stages
    eps_decay_fraction = 0.25  # Decay over 1/4 of total training run
    eps_decay_steps = int(args.train_steps * eps_decay_fraction)  # Total steps for epsilon decay
    
    # Track previous stage to detect stage transitions
    prev_stage_num = 0
    
    for step in range(1, args.train_steps + 1):
        # Determine current stage (1 = Bird Training, 2 = Pipe Training)
        if use_stages:
            stage_num = ((step - 1) // stage_duration) % 2 + 1
            bird_training_stage = (stage_num == 1)
            pipe_training_stage = (stage_num == 2)
            
            # Detect stage transition
            if stage_num != prev_stage_num:
                if bird_training_stage:
                    # Entering bird training stage: set epsilon based on global progress
                    # Epsilon decays from 0.2 to 0.05 over first 1/4 of total training
                    if step <= eps_decay_steps:
                        # Still in decay period: calculate epsilon based on global step
                        frac = step / float(eps_decay_steps)
                        bird_agent.eps = bird_stage_eps_start + frac * (bird_stage_eps_end - bird_stage_eps_start)
                    else:
                        # After decay period: use final epsilon
                        bird_agent.eps = bird_stage_eps_end
                    bird_agent.steps = 0  # Reset step counter
                    print(f"\n[Stage Transition] Entering Bird Training Stage (step {step}) - Epsilon set to {bird_agent.eps:.3f}")
                elif pipe_training_stage:
                    # Entering pipe training stage: force epsilon to 0 (greedy)
                    bird_agent.eps = 0.0
                    print(f"\n[Stage Transition] Entering Pipe Training Stage (step {step}) - Epsilon set to 0.0 (greedy)")
                prev_stage_num = stage_num
            
            # During bird training stage: update epsilon based on global progress
            if bird_training_stage:
                if step <= eps_decay_steps:
                    # Still in decay period: decay epsilon linearly from 0.2 to 0.05
                    frac = step / float(eps_decay_steps)
                    bird_agent.eps = bird_stage_eps_start + frac * (bird_stage_eps_end - bird_stage_eps_start)
                else:
                    # After decay period: keep epsilon at 0.05
                    bird_agent.eps = bird_stage_eps_end
            
            # During pipe training stage, keep epsilon at 0
            if pipe_training_stage:
                bird_agent.eps = 0.0
        else:
            # Simultaneous training: both train
            bird_training_stage = True
            pipe_training_stage = True
        
        # Bird action selection
        # Track if action was random exploration (for filtering pipe rewards)
        action_was_random = False
        if step < args.warmup_steps:
            a = int(np.random.randint(0, action_dim))
            action_was_random = True
        else:
            # Use greedy action if in greedy mode OR if we're in pipe training stage
            if args.greedy_mode or (use_stages and not bird_training_stage):
                a = bird_agent.greedy_action(s)
            else:
                # Use epsilon-greedy with exploration flag to track random actions
                # During bird training stage, epsilon is controlled manually (skip internal update)
                skip_eps_update = use_stages and bird_training_stage
                a, action_was_random = bird_agent.epsilon_greedy_with_flag(s, skip_epsilon_update=skip_eps_update)

        # Apply pipe controls before step
        # Use greedy mode for pipes if we're in bird training stage
        pipe_greedy = use_stages and bird_training_stage
        nn_pipes_before = apply_pipe_controls(env, pipe_controllers, pipe_greedy=pipe_greedy)  # Store state/action BEFORE step

        # Frame skip: repeat the same action for N steps, accumulate reward
        total_r = 0.0
        for _ in range(max(1, args.frameskip)):
            s2, r, done, info = env.step(a)
            
            # Re-apply pipe controls after step (pipes move continuously)
            apply_pipe_controls(env, pipe_controllers, pipe_greedy=pipe_greedy)  # Don't overwrite nn_pipes_before
            
            info["epsilon"] = bird_agent.eps
            if args.render:
                env.draw_hud(info)
            total_r += r
            if done:
                break

        # Store bird experience (only if bird is training)
        if bird_training_stage or not use_stages:
            bird_buf.push(s, a, total_r, s2, done)

        # Check if bird passed a pipe (score increased)
        current_score = info.get("score", 0)
        pipe_passed = (current_score > prev_score)
        prev_score = current_score

        if pipe_training_stage or not use_stages:
            for pipe_idx, pipe_data in nn_pipes_before.items():
                if pipe_idx < len(env.pipes.upper):
                    next_state = _extract_pipe_controller_state(env, pipe_idx)
                    if next_state is not None:
                        pipe_reward = 0.0
                        
                        if pipe_passed and pipe_idx == 0:
                            # Bird passed pipe_idx 0 (closest pipe) - calculate proximity reward
                            low = env.pipes.lower[0]
                            gap_center_y = low.y - env.pipes.pipe_gap / 2.0
                            gap_half = env.pipes.pipe_gap / 2.0
                            
                            bird_y = env.player.y
                            upper_edge = gap_center_y - gap_half
                            lower_edge = gap_center_y + gap_half
                            
                            # Distance to nearest edge
                            dist_to_upper = abs(bird_y - upper_edge)
                            dist_to_lower = abs(bird_y - lower_edge)
                            min_dist_to_edge = min(dist_to_upper, dist_to_lower)
                            
                            # Normalize distance (0 = at edge, gap_half = at center)
                            # Closer to edge = better for pipes
                            proximity_score = 1.0 - (min_dist_to_edge / gap_half)
                            proximity_score = max(0.0, min(1.0, proximity_score))  # Clamp [0, 1]
                            
                            # Reward: -1.0 (base penalty) + proximity bonus (up to +0.5)
                            # If bird barely passed (proximity_score ~1.0), pipes get -0.5 instead of -1.0
                            proximity_bonus = proximity_score * 0.5
                            pipe_reward = -1.0 + proximity_bonus
                        elif total_r < -0.5 and not action_was_random:
                            # Bird died - higher reward for pipes
                            pipe_reward = 2.0  # Increased from 1.0 to 2.0
                        
                        pipe_controllers.push_experience(
                            pipe_idx, pipe_data["state"], pipe_data["action"],
                            pipe_reward, next_state, done
                        )

        s = s2
        ep_reward += total_r

        # Train bird DQN (only in bird training stage)
        bird_loss = None
        if (bird_training_stage or not use_stages) and step >= args.warmup_steps and len(bird_buf) >= args.batch_size and step % args.optimize_every == 0:
            batch = bird_buf.sample(args.batch_size)
            bird_loss = bird_agent.optimize(batch)

        # Train pipe controllers (only in pipe training stage)
        pipe_loss = None
        if (pipe_training_stage or not use_stages) and step >= args.pipe_warmup_steps and step % args.pipe_optimize_every == 0:
            pipe_loss = pipe_controllers.optimize()
            if pipe_loss is not None:
                last_pipe_loss = pipe_loss  # Track last valid loss

        if done:
            scores.append(info.get("score", 0))
            score_steps.append(step)  # Track step when episode ended
            s = env.reset()
            prev_score = 0  # Reset score tracking for new episode
            episode += 1
            ep_reward = 0.0

        # Early stopping
        if args.max_seconds > 0 and (time.time() - t0) >= args.max_seconds:
            break
        if args.max_episodes > 0 and episode > args.max_episodes:
            break

        # Logging
        if step % args.log_every == 0:
            ma10 = np.mean(scores[-10:]) if scores else 0.0
            ma10_pipes = int(ma10) if scores else 0
            
            # Calculate MA for last 1k steps
            ma1k_pipes = 0
            if score_steps:
                # Find scores from episodes that ended in the last 1k steps
                last_1k_scores = [scores[i] for i, s_step in enumerate(score_steps) if s_step >= step - 1000]
                if last_1k_scores:
                    ma1k = np.mean(last_1k_scores)
                    ma1k_pipes = int(ma1k)
            
            bird_loss_str = f"{bird_loss:.4f}" if bird_loss is not None else "-"
            # Use last_pipe_loss if current pipe_loss is None (to show recent training)
            pipe_loss_to_show = pipe_loss if pipe_loss is not None else last_pipe_loss
            pipe_loss_str = f"{pipe_loss_to_show:.4f}" if pipe_loss_to_show is not None else "-"
            
            # Show current stage
            stage_info = ""
            if use_stages:
                stage_name = "Bird Training" if bird_training_stage else "Pipe Training"
                stage_info = f" [{stage_name}]"
            
            print(
                f"step={step:6d} ep={episode:4d} eps={bird_agent.eps:.3f}{stage_info} "
                f"ma10={ma10_pipes} ma1k={ma1k_pipes} pipes | "
                f"bird_loss={bird_loss_str} pipe_loss={pipe_loss_str}"
            )

            if args.viz and scores:
                try:
                    import matplotlib.pyplot as plt

                    fig = plt.figure(figsize=(10, 4))
                    ax1 = fig.add_subplot(121)
                    ax1.plot(scores, label="score", alpha=0.5)
                    if len(scores) >= 10:
                        ma = np.convolve(scores, np.ones(10) / 10, mode="valid")
                        ax1.plot(range(9, 9 + len(ma)), ma, label="ma10")
                    ax1.legend()
                    ax1.set_title("Bird Score")
                    ax1.set_xlabel("Episode")
                    ax1.set_ylabel("Score")
                    
                    ax2 = fig.add_subplot(122)
                    ax2.set_title("Training Progress")
                    ax2.text(0.1, 0.5, f"Steps: {step}\nEpisodes: {episode}\nMA10: {ma10:.2f}", 
                            transform=ax2.transAxes, fontsize=12)
                    ax2.axis('off')
                    
                    fig.tight_layout()
                    out = checkpoints / "progress.png"
                    fig.savefig(out)
                    plt.close(fig)
                except Exception:
                    pass

        # Checkpointing
        if step % args.ckpt_every == 0:
            # Save bird agent
            bird_ckpt_path = checkpoints / f"bird_ckpt_step_{step}.pt"
            bird_agent.save(str(bird_ckpt_path))

            ma10 = np.mean(scores[-10:]) if scores else 0.0
            ma10_pipes = int(ma10) if scores else 0
            if ma10 > best_ma:
                best_ma = ma10
                bird_agent.save(str(checkpoints / "best_bird.pt"))
                print(f"✓ New best bird agent (ma10={ma10_pipes} pipes)")

            # Save pipe controllers
            pipe_controllers.save(str(pipe_checkpoints))
            print(f"✓ Saved checkpoints at step {step}")

    env.close()

    # Final save
    print("\n" + "="*60)
    print("Training complete!")
    final_ma10 = np.mean(scores[-10:]) if scores else 0.0
    final_ma10_pipes = int(final_ma10) if scores else 0
    best_ma_pipes = int(best_ma) if best_ma > -1e9 else 0
    print(f"Final MA10: {final_ma10_pipes} pipes (avg of last 10 runs)")
    print(f"Best MA10: {best_ma_pipes} pipes")
    print("="*60)
    
    bird_agent.save(str(checkpoints / "final_bird.pt"))
    pipe_controllers.save(str(pipe_checkpoints))
    print("✓ Saved final checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Co-train bird DQN and adversarial pipe controllers"
    )
    
    # Environment
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--mute", dest="mute", action="store_true")
    parser.add_argument("--no-mute", dest="mute", action="store_false")
    parser.set_defaults(mute=True)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--flap-cost", type=float, default=0.003)
    parser.add_argument("--out-of-bounds-cost", type=float, default=0.005)
    parser.add_argument("--center-reward", type=float, default=0.0)

    # Bird DQN
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3, help="Bird DQN learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=int, default=90_000, 
                        help="Epsilon decay steps (default 90k for 100k training steps)")
    parser.add_argument("--grad-clip", type=float, default=5.0)

    # Pipe Controllers
    parser.add_argument("--pipe-lr", type=float, default=1e-3, help="Pipe controller learning rate")
    parser.add_argument("--pipe-batch-size", type=int, default=64)
    parser.add_argument("--pipe-capacity", type=int, default=50_000)
    parser.add_argument("--pipe-grad-clip", type=float, default=5.0)
    parser.add_argument("--pipe-warmup-steps", type=int, default=20_000, help="Steps before pipe training starts (let bird learn first)")
    parser.add_argument("--pipe-optimize-every", type=int, default=4, help="Train pipes every N steps")

    # Replay
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--optimize-every", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)

    # Training
    parser.add_argument("--train-steps", type=int, default=300_000)
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    
    # Resume
    parser.add_argument("--resume-bird", type=str, default="", help="Path to bird checkpoint")
    parser.add_argument("--resume-pipes", type=str, default="", help="Path to pipe controller checkpoint directory")
    parser.add_argument("--eps-start-override", type=float, default=None, help="Override epsilon start value when resuming (e.g., 0.5)")
    parser.add_argument("--greedy-mode", action="store_true", help="Use greedy policy (epsilon=0) for well-trained bird")
    
    # Stage-based training
    parser.add_argument("--stage-duration", type=int, default=0, 
                        help="Steps per stage (0 = simultaneous training). Stage 1: Bird trains, Pipes greedy. Stage 2: Bird greedy, Pipes train.")
    
    # Visualization
    parser.add_argument("--viz", action="store_true")

    args = parser.parse_args()
    train_adversarial(args)


if __name__ == "__main__":
    main()

