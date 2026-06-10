import argparse
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


def set_cube_mass(env, mass):
    """Override the cube mass on the underlying PyBullet env (after reset)."""
    sim = env.unwrapped.task.sim
    object_body_id = sim._bodies_idx["object"]
    sim.physics_client.changeDynamics(
        bodyUniqueId=object_body_id, linkIndex=-1, mass=float(mass)
    )


def evaluate(model_path: str, n_episodes: int, deterministic: bool, render: bool,
             env_type: str, algo: str, mass: float = None) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    set_seed(SEED)

    render_mode = "human" if render else "rgb_array"

    if algo == "ppo":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.evaluation import evaluate_policy

        def make_env():
            e = gym.make("PandaPush-v3", render_mode=render_mode,
                         type=env_type, reward_type="dense")
            e.action_space.seed(SEED)
            return e

        env = DummyVecEnv([make_env])
        env.seed(SEED)

        vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
            print(f"VecNormalize stats loaded from {vecnorm_path}")
        else:
            print(f"WARNING: {vecnorm_path} not found. PPO trained WITH VecNormalize, "
                  f"so results without it are WRONG/identical.")

        model = PPO.load(model_path, env=env)

        if mass is not None:
            env.reset()
            set_cube_mass(env.envs[0], mass)
            print(f"Cube mass overridden to {mass} kg")

        successes = []

        def success_cb(locals_, globals_):
            if locals_["done"]:
                info = locals_["info"]
                if isinstance(info, dict) and "is_success" in info:
                    successes.append(float(info["is_success"]))

        episode_returns, _ = evaluate_policy(
            model, env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            callback=success_cb,
            return_episode_rewards=True,
        )
        env.close()

        returns = np.array(episode_returns, dtype=np.float32)
        for i, r in enumerate(returns, start=1):
            print(f"Episode {i:03d} | return = {r:.3f}")

        print("\n=== Evaluation summary ===")
        print(f"Algorithm:    PPO")
        print(f"Env type:     {env_type}")
        print(f"Cube mass:    {mass if mass is not None else 'default'}")
        print(f"Eval seed:    {SEED}")
        print(f"Episodes:     {n_episodes}")
        print(f"Mean return:  {returns.mean():.3f}")
        print(f"Std return:   {returns.std():.3f}")
        print(f"Min return:   {returns.min():.3f}")
        print(f"Max return:   {returns.max():.3f}")
        if successes:
            print(f"Success rate: {np.mean(successes):.2%}")
        return

    elif algo == "sac":
        from stable_baselines3 import SAC
        env = gym.make("PandaPush-v3", render_mode=render_mode,
                       type=env_type, reward_type="dense")
        env.action_space.seed(SEED)
        model = SAC.load(model_path)

    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=SEED if episode == 1 else None)

        if mass is not None:
            set_cube_mass(env, mass)

        terminated = False
        truncated = False
        episode_return = 0.0
        step_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            step_info = info

        episode_returns.append(episode_return)

        if isinstance(step_info, dict) and "is_success" in step_info:
            successes.append(float(step_info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Algorithm:    {algo.upper()}")
    print(f"Env type:     {env_type}")
    print(f"Cube mass:    {mass if mass is not None else 'default'}")
    print(f"Eval seed:    {SEED}")
    print(f"Episodes:     {n_episodes}")
    print(f"Mean return:  {returns.mean():.3f}")
    print(f"Std return:   {returns.std():.3f}")
    print(f"Min return:   {returns.min():.3f}")
    print(f"Max return:   {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO or SAC on PandaPush-v3")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to a model zip file (e.g., ppo_push.zip)")
    parser.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"],
                        help="Algorithm to load (ppo or sac)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of eval episodes")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy sampling instead of deterministic actions")
    parser.add_argument("--render", action="store_true",
                        help="Render with a window (render_mode='human')")
    parser.add_argument("--env-type", type=str, default="target", choices=["source", "target"],
                        help="Type of environment to evaluate on (default: target)")
    parser.add_argument("--mass", type=float, default=None,
                        help="Override cube mass (kg). If omitted, uses the env's default mass.")
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
        mass=args.mass,
    )