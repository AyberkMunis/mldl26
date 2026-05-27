import gymnasium as gym
import argparse
import torch
import time
from agent import Policy, Agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Ortamı görselleştirme (render) moduyla oluşturuyoruz
    env = gym.make('Hopper-v4', render_mode='human')
    
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    
    policy = Policy(state_space, action_space)
    
    # Kaydedilen modeli yüklüyoruz
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    print(f"\nYüklenen Model: {checkpoint['algo']}")
    print(f"Eğitim sırasında kaydedildiği episode: {checkpoint['episode']}")
    print(f"Eğitim sırasında ulaştığı en iyi ödül: {checkpoint['reward']:.2f}\n")

    agent = Agent(policy=policy, device='cpu')

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            # Test sırasında rastgeleliği kapatıp doğrudan modelin öğrendiği ana davranışı (mean) alıyoruz (evaluation=True)
            action, _ = agent.get_action(state, evaluation=True)
            
            state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated
            total_reward += reward
            env.render()
            
        print(f"Test Episode {episode} - Toplam Ödül: {total_reward:.2f}")

    env.close()

if __name__ == '__main__':
    main()
