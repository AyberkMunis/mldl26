import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.monitor import Monitor
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO or SAC on PandaPush-v3")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
    )
    parser.add_argument(
        "--env-type", type=str, default="source", choices=["source", "target"]
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--log-interval", type=int, default=20)
    # --- wandb args ---
    parser.add_argument("--wandb-project", type=str, default="MLDL")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.wandb_run_name or (
        f"{args.algo}_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=vars(args),
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )
    env = RandomizationWrapper(env, mass_range=(0.5, 6.0), mode=args.sampling_strategy)
    env = Monitor(env)

    tensorboard_log = f"runs/{run.id}"

    if args.algo == "ppo":
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model_hyperparams = dict(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )
        model = PPO("MultiInputPolicy", env, **model_hyperparams)

    elif args.algo == "sac":
        policy_kwargs = dict(net_arch=[512, 512])
        model_hyperparams = dict(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )
        model = SAC("MultiInputPolicy", env, **model_hyperparams)

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # >>> Sadece modele verilen hiperparametreleri wandb'ye yaz <
    wandb.config.update({"model": model_hyperparams}, allow_val_change=True)

    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        model_save_freq=50_000,
        gradient_save_freq=10_000,
        verbose=2,
    )

    model.learn(
        total_timesteps=args.timesteps,
        log_interval=args.log_interval,
        callback=wandb_callback,
    )

    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model.save(save_name)
    print(f"Model saved successfully as {save_name}.zip")

    artifact = wandb.Artifact(name=save_name, type="model")
    artifact.add_file(f"{save_name}.zip")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    main()