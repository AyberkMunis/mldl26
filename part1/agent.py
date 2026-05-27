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

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.log_std = torch.nn.Parameter(torch.zeros(self.action_space))


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
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
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.tanh(self.fc3_actor_mean(x_actor))

        log_std = torch.clamp(self.log_std, -5, 1)
        sigma = log_std.exp()
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network

        
        return normal_dist

    def value(self, x):
        """
        Critic forward pass: estimates V(s)
        """
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

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #


        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        if self.algo == 'reinforce':
            # REINFORCE without baseline
            returns = discount_rewards(rewards, self.gamma, done)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            policy_loss = -(action_log_probs * returns.detach()).mean()
            loss = policy_loss

        elif self.algo == 'reinforce_baseline':
            # REINFORCE with constant baseline
            returns = discount_rewards(rewards, self.gamma, done)

            # Constant baseline: A_t = G_t - b
            advantages = returns - self.baseline
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(action_log_probs * advantages.detach()).mean()
            loss = policy_loss

        elif self.algo == 'actor_critic':
            # Actor-Critic: Monte Carlo returns with learned value baseline
            values = self.policy.value(states)

            # Compute full episode returns
            returns = discount_rewards(rewards, self.gamma, done)
            
            # CRITICAL FIX: Normalize returns to prevent Critic Loss explosion!
            # Critic will now predict normalized values (around zero) instead of huge rewards (2000+)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Advantage = G_t - V(s_t)
            with torch.no_grad():
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(action_log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns.detach())

            # 0.5 is standard weighting for critic loss
            loss = actor_loss + 0.5 * critic_loss

        else:
            raise ValueError(f"Unknown algorithm: {self.algo}")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return loss.item()


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            action= torch.clamp(normal_dist.mean, -1.0,1.0)
            return action, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()
            action=torch.clamp(action,-1,1)

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

