
import argparse
import random
from collections import deque
 
import gymnasium as gym
import numpy as np
import torch
import panda_gym  # type: ignore[import-not-found]
import wandb
 
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
    parser.add_argument("--seed", type=int, default=42)
    # --- wandb args ---
    parser.add_argument("--wandb-project", type=str, default="part2")
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
    seed = args.seed
 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
    run_name = args.wandb_run_name or (
        f"{args.algo}_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k_train_{seed}"
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
 
    def make_env():
        env = gym.make(
            "PandaPush-v3",
            render_mode="rgb_array",
            type=args.env_type,
            reward_type="dense",
        )
        env.reset(seed=seed)
        env = RandomizationWrapper(env, mass_range=(0.5, 6.0), mode=args.sampling_strategy)
        env = Monitor(env)
        return env
 
    tensorboard_log = f"runs/{run.id}"
 
    # VecNormalize is applied ONLY to PPO. PPO is highly sensitive to obs/reward
    # scaling and collapses without it. SAC is left raw: reward normalization with
    # an off-policy replay buffer mixes stale statistics and is not standard.
 
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
            seed=seed,
            device="cuda"
        )
        model = PPO("MultiInputPolicy", env, **model_hyperparams)
 
    elif args.algo == "sac":
      env = make_env()
 
      policy_kwargs = dict(net_arch=[512, 512])

      model_hyperparams = dict(
          learning_rate=1e-4,        # 3e-4'ten düşürüldü: Q kararsızlığını azaltır
          buffer_size=1_000_000,     # Colab'da OOM olursa 300_000'e çek
          batch_size=256,
          tau=0.005,
          gamma=0.99,
          learning_starts=10_000,    # erken Q patlamasının ana ilacı
          train_freq=1,
          gradient_steps=1,
          policy_kwargs=policy_kwargs,
          verbose=1,
          tensorboard_log=tensorboard_log,
          seed=seed,
          device="cuda"
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
 
    clean_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k_{seed}"

    # 1) Önce hızlı/güvenilir lokal diske kaydet
    local_path = f"/content/{clean_name}.zip"
    model.save(local_path)
    print(f"Model saved locally as {local_path}")

    # 2) wandb artifact'ı LOKAL dosyadan oluştur (isimde slash yok)
    artifact = wandb.Artifact(name=clean_name, type="model")
    artifact.add_file(local_path)

    run.log_artifact(artifact)

    # 3) Drive'a kopyala (klasörü garanti et)
    drive_dir = "/content/drive/MyDrive/Mldl_project/part2"
    os.makedirs(drive_dir, exist_ok=True)
    shutil.copy(local_path, f"{drive_dir}/{clean_name}.zip")
    print(f"Copied to Drive: {drive_dir}/{clean_name}.zip")
 
 
if __name__ == "__main__":
    main()
 