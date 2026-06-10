import argparse
import os
import glob
import random

import gymnasium as gym
import numpy as np
import torch
import pandas as pd
import panda_gym  # noqa: F401

SEED = 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_cube_mass(env, mass):
    sim = env.unwrapped.task.sim
    object_body_id = sim._bodies_idx["object"]
    sim.physics_client.changeDynamics(
        bodyUniqueId=object_body_id, linkIndex=-1, mass=float(mass)
    )


def evaluate_model_at_mass(model_path, n_episodes, deterministic, mass, algo):
    set_seed(SEED)

    if algo == "ppo":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.evaluation import evaluate_policy

        def make_env():
            e = gym.make("PandaPush-v3", render_mode="rgb_array",
                         type="source", reward_type="dense")
            e.action_space.seed(SEED)
            return e

        env = DummyVecEnv([make_env])
        vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False

        model = PPO.load(model_path, env=env)

        env.reset()
        set_cube_mass(env.envs[0], mass)

        successes = []
        def cb(locals_, globals_):
            if locals_["done"]:
                info = locals_["info"]
                if isinstance(info, dict) and "is_success" in info:
                    successes.append(float(info["is_success"]))

        episode_returns, _ = evaluate_policy(
            model, env, n_eval_episodes=n_episodes,
            deterministic=deterministic, callback=cb,
            return_episode_rewards=True,
        )
        env.close()
        returns = np.array(episode_returns, dtype=np.float32)
        success_rate = float(np.mean(successes)) if successes else None

    else:  # sac
        from stable_baselines3 import SAC
        env = gym.make("PandaPush-v3", render_mode="rgb_array",
                       type="source", reward_type="dense")
        env.action_space.seed(SEED)
        model = SAC.load(model_path)

        episode_returns = []
        successes = []
        for episode in range(1, n_episodes + 1):
            obs, _ = env.reset(seed=SEED if episode == 1 else None)
            set_cube_mass(env, mass)   # reset'ten SONRA kütleyi set et
            terminated = truncated = False
            ep_ret = 0.0
            step_info = {}
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                step_info = info
            episode_returns.append(ep_ret)
            if isinstance(step_info, dict) and "is_success" in step_info:
                successes.append(float(step_info["is_success"]))
        env.close()
        returns = np.array(episode_returns, dtype=np.float32)
        success_rate = float(np.mean(successes)) if successes else None

    return {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "success_rate": success_rate,
    }


def parse_name(path):
    """sac_push_none_source_500k_42.zip -> (algo, sampling, train_env, timestep, seed)"""
    base = os.path.basename(path).replace(".zip", "")
    parts = base.split("_")
    algo = parts[0]
    sampling = parts[2]
    train_env = parts[3]
    timestep = parts[4]
    seed = int(parts[5])
    return algo, sampling, train_env, timestep, seed


def main():
    ap = argparse.ArgumentParser(description="Mass-sweep eval tüm modeller -> Excel")
    ap.add_argument("--models-dir", type=str, required=True,
                    help="Model .zip dosyalarının olduğu klasör")
    ap.add_argument("--output", type=str, default="mass_sweep_results.xlsx")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--masses", type=float, nargs="+",
                    default=[1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                    help="Test edilecek küp kütleleri (kg)")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    deterministic = not args.stochastic
    model_files = sorted(glob.glob(os.path.join(args.models_dir, "*.zip")))
    if not model_files:
        raise FileNotFoundError(f"{args.models_dir} içinde .zip model bulunamadı.")

    rows = []
    for mp in model_files:
        try:
            algo, sampling, train_env, timestep, seed = parse_name(mp)
        except (IndexError, ValueError):
            print(f"İsim parse edilemedi, atlanıyor: {mp}")
            continue

        for mass in args.masses:
            print(f"Eval: {os.path.basename(mp)} | mass={mass} ...")
            try:
                m = evaluate_model_at_mass(mp, args.episodes, deterministic, mass, algo)
            except Exception as e:
                print(f"  HATA ({e}), atlanıyor.")
                continue

            rows.append({
                "Algoritma": algo.upper(),
                "Sampling": sampling,
                "Time Step": timestep,
                "Seed": seed,
                "Train Env-Type": train_env,
                "Test Mass": mass,
                "Mean Return": round(m["mean"], 3),
                "Std Return": round(m["std"], 3),
                "Min Return": round(m["min"], 3),
                "Max Return": round(m["max"], 3),
                "Success Rate": round(m["success_rate"], 4) if m["success_rate"] is not None else None,
                "test seed": SEED,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["Algoritma", "Sampling", "Train Env-Type", "Seed", "Test Mass"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
        ws = writer.sheets["results"]
        sr_col = df.columns.get_loc("Success Rate") + 1
        for row in range(2, len(df) + 2):
            ws.cell(row=row, column=sr_col).number_format = "0%"

        pivot = df.pivot_table(
            index=["Algoritma", "Sampling", "Train Env-Type", "Seed"],
            columns="Test Mass", values="Success Rate", aggfunc="mean",
        ).reset_index()
        pivot.to_excel(writer, index=False, sheet_name="success_pivot")

    print(f"\nKaydedildi: {args.output}  ({len(df)} satır)")


if __name__ == "__main__":
    main()