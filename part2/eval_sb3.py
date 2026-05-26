import argparse
import os

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401 - required so Panda envs are registered


def evaluate(model_path: str, n_episodes: int, deterministic: bool, render: bool, env_type: str, algo: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    render_mode = "human" if render else "rgb_array"
    env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")
    
    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    elif algo == "sac":
        from stable_baselines3 import SAC
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    episode_returns = []
    successes = []
    successful_steps = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0
        step_count = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            step_count += 1
            if render:
                import time
                time.sleep(0.05)  # Slow down the visualization to calmly observe robot dynamics


        episode_returns.append(episode_return)

        is_success = False
        if isinstance(info, dict) and "is_success" in info:
            is_success = bool(info["is_success"])
            successes.append(float(is_success))

        if is_success:
            successful_steps.append(step_count)
            status_str = f"SUCCESS (in {step_count} steps)"
        else:
            status_str = f"FAILED (limit {step_count} steps)"

        print(f"Episode {episode:03d} | return = {episode_return:.3f} | Result: {status_str}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate (Accuracy): {success_rate:.2%}")

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
        default=500, 
        help="Number of eval episodes"
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

