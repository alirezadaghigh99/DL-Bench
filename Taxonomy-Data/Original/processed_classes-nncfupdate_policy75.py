    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(
            self.batch_size
        )

        # normalize the reward
        batch_mean_reward = np.mean(reward_batch)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(
                [
                    to_tensor(next_state_batch),
                    self.actor_target(to_tensor(next_state_batch)),
                ]
            )

        target_q_batch = (
            to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(float)) * next_q_values
        )

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = (-1) * self.critic([to_tensor(state_batch), self.actor(to_tensor(state_batch))])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        # update for log
        self.value_loss = value_loss
        self.policy_loss = policy_loss