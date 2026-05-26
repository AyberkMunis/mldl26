"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import argparse
import os
import time
import numpy as np
import torch

from agent import Policy,Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="reinforce",
        choices=["reinforce", "reinforce_baseline", "actor_critic"]
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--baseline", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(state_space, action_space)
    agent = Agent(
        policy=policy,
        algo=args.algo,
        baseline=args.baseline,
        gamma=args.gamma,
        lr=args.lr
    )

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{args.algo}_best.pt"

    best_reward = -float("inf")
    episode_rewards = []
    start_time = time.time()

    #TODO: implement training loop for REINFORCE and Actor-Critic using the agent defined in agent.py
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action, action_log_prob = agent.get_action(state, evaluation=False)

            next_state, reward, terminated, truncated, _ = env.step(
                action.detach().cpu().numpy()
            )
            done = terminated or truncated

            agent.store_outcome(
                state=state,
                next_state=next_state,
                action_log_prob=action_log_prob,
                reward=reward,
                done=done
            )

            state = next_state
            total_reward += reward

        loss = agent.update_policy()
        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "algo": args.algo,
                    "episode": episode,
                    "reward": best_reward,
                },
                checkpoint_path,
            )

        if episode % 10 == 0:
            avg_last_10 = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode:5d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Avg last 10: {avg_last_10:8.2f} | "
                f"Best: {best_reward:8.2f} | "
                f"Loss: {loss:8.4f}"
            )

    elapsed = time.time() - start_time
    print("\nTraining finished.")
    print("Algorithm:", args.algo)
    print(f"Best training reward: {best_reward:.2f}")
    print(f"Training time: {elapsed:.2f} seconds")

    env.close()

if __name__ == '__main__':
    main()