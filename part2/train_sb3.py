import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO, SAC, DDPG
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO or SAC on PandaPush-v3")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="RL algorithm to use (ppo or sac)",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Log interval for stable-baselines3 training (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    # Use a wider range (0.5, 6.0) that encloses the target mass (5.0 kg)
    env = RandomizationWrapper(env, mass_range=(0.5, 6.0), mode=args.sampling_strategy)

    if args.algo == "ppo":
        # Optimized PPO Hyperparameters for PandaPush-v3
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    elif args.algo == "sac":
        # Optimized SAC Hyperparameters for PandaPush-v3 (larger network capacity)
        policy_kwargs = dict(net_arch=[512, 512])
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")


    model.learn(total_timesteps=args.timesteps, log_interval=args.log_interval)

    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model.save(save_name)
    print(f"Model saved successfully as {save_name}.zip")



if __name__ == "__main__":
    main()