import argparse
import math
import os
import random

import gymnasium as gym
import numpy as np
import torch
import panda_gym  # noqa: F401 - required so Panda envs are registered

SEED = 200


def set_seed(seed: int) -> None:
    """Seed all RNGs so evaluation is reproducible across runs/models."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model_path: str, n_episodes: int, deterministic: bool, render: bool, env_type: str, algo: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    # Global determinism for the whole evaluation run.
    set_seed(SEED)

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    render_mode = "human" if render else "rgb_array"
    base_env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")
    base_env.action_space.seed(SEED)

    use_vecenv = False

    if algo == "ppo":
        from stable_baselines3 import PPO
        vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
        env = DummyVecEnv([lambda: base_env])
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
            use_vecenv = True
            print(f"VecNormalize stats loaded from {vecnorm_path}")
        else:
            print(f"Warning: {vecnorm_path} not found, running without VecNormalize.")
        model = PPO.load(model_path, env=env)
    elif algo == "sac":
        from stable_baselines3 import SAC
        env = base_env
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    episode_returns = []
    successes = []
    successful_steps = []

    for episode in range(1, n_episodes + 1):
        if use_vecenv:
            obs, _ = env.reset(seed=[SEED] if episode == 1 else None)
        else:
            obs, _ = env.reset(seed=SEED if episode == 1 else None)

        done = False
        episode_return = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)

            if use_vecenv:
                obs, reward, terminated, info = env.step(action)
                done = bool(terminated[0])
                episode_return += float(reward[0])
                step_count += 1
                step_info = info[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_return += float(reward)
                step_count += 1
                if render:
                    import time
                    time.sleep(0.05)
                step_info = info

        episode_returns.append(episode_return)

        if "is_success" not in step_info:
            raise KeyError("Expected 'is_success' in info dict but it was missing.")
        is_success = bool(step_info["is_success"])
        successes.append(float(is_success))

        if is_success:
            successful_steps.append(step_count)
            status_str = f"SUCCESS (in {step_count} steps)"
        else:
            status_str = "FAILED"

        print(f"Episode {episode:03d} | return = {episode_return:.3f} | Result: {status_str}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Algorithm:   {algo.upper()}")
    print(f"Env type:    {env_type}")
    print(f"Eval seed:   {SEED}")
    print(f"Episodes:    {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        # Standard error of a Bernoulli proportion: sqrt(p(1-p)/n).
        # Reported so small differences aren't over-interpreted at n=50.
        se = math.sqrt(success_rate * (1.0 - success_rate) / n_episodes)
        print(f"Success rate (Accuracy): {success_rate:.2%}  (+/- {se:.2%} SE)")

    if successful_steps:
        print(f"Average steps to success: {np.mean(successful_steps):.1f} steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC or PPO on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a model zip file (e.g., ppo_panda_push.zip)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="Algorithm to load (ppo or sac)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of eval episodes",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--env-type",
        type=str, default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on (default: target)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
        algo=args.algo,
    )