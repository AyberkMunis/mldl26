import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma, done):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        if done[t]:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 256
        self.tanh = torch.nn.Tanh()

      
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        self.log_std = torch.nn.Parameter(torch.zeros(self.action_space))



       
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
       
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.tanh(self.fc3_actor_mean(x_actor))

        log_std = torch.clamp(self.log_std, -5, 1)
        sigma = log_std.exp()
        normal_dist = Normal(action_mean, sigma)



        
        return normal_dist

    def value(self, x):
     
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic).squeeze(-1)
        return state_value


class Agent(object):
    def __init__(self, policy, device='cpu', algo='reinforce', baseline=0.0, gamma=0.99, lr=1e-4):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        
        actor_params = [p for n, p in policy.named_parameters() if 'critic' not in n]
        critic_params = [p for n, p in policy.named_parameters() if 'critic' in n]
        
        self.optimizer = torch.optim.Adam([
            {'params': actor_params, 'lr': lr},
            {'params': critic_params, 'lr': 1e-3}
        ])

        self.algo = algo
        self.baseline = baseline
        self.gamma = gamma
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        
        if self.algo == 'reinforce':
            returns = discount_rewards(rewards, self.gamma, done)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            policy_loss = -(action_log_probs * returns.detach()).mean()
            loss = policy_loss




        elif self.algo == 'reinforce_baseline':
            returns = discount_rewards(rewards, self.gamma, done)

            advantages = returns - self.baseline
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(action_log_probs * advantages.detach()).mean()
            loss = policy_loss

        elif self.algo == 'actor_critic':
            
            values = self.policy.value(states)

            returns = discount_rewards(rewards, self.gamma, done)
            
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            with torch.no_grad():
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(action_log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns.detach())

            loss = actor_loss + 0.5 * critic_loss


        else:
            raise ValueError(f"Unknown algorithm: {self.algo}")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.last_actor_loss = actor_loss.item() if 'actor_loss' in locals() else policy_loss.item()
        self.last_critic_loss = critic_loss.item() if 'critic_loss' in locals() else 0.0
        return loss.item()


    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation: 
            action= torch.clamp(normal_dist.mean, -1.0,1.0)
            return action, None

        else:  
            action = normal_dist.sample()

            action_log_prob = normal_dist.log_prob(action).sum()
            action=torch.clamp(action,-1,1)

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

