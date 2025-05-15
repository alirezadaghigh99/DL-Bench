import torch
import torch.nn.functional as F

class DDPG:
    def __init__(self, nb_states, nb_actions, iter_number: int = None, hparam_override: dict = None):
        self.memory = None  # replay buffer
        self.actor = None  # actor network
        self.actor_target = None  # target actor network
        self.actor_optim = None  # optimizer for actor network
        self.critic = None  # critic network
        self.critic_target = None  # target critic network
        self.critic_optim = None  # optimizer for critic network
        self.batch_size = None  # batch size for training
        self.discount = None  # discount factor
        self.moving_average = 0.0  # moving average of rewards
        self.moving_alpha = 0.01  # smoothing factor for moving average
        self.value_loss = 0.0  # loss for critic network
        self.policy_loss = 0.0  # loss for actor network

    def update_policy(self):
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Normalize rewards
        self.moving_average = self.moving_alpha * rewards.mean() + (1 - self.moving_alpha) * self.moving_average
        rewards = rewards - self.moving_average

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.discount * target_q_values

        # Update critic network
        predicted_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(predicted_q_values, target_q_values)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update actor network
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update target networks
        tau = 0.005  # Soft update factor
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Update internal attributes for logging
        self.value_loss = critic_loss.item()
        self.policy_loss = actor_loss.item()