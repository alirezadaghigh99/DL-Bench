    def select_action(self, s_t, episode, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array(s_t).reshape(1, -1)))).squeeze(0)

        if decay_epsilon is True:
            self.delta = self.init_delta * (self.delta_decay ** (episode - self.warmup_iter_number))
            action = sample_from_truncated_normal_distribution(
                lower=self.LBOUND, upper=self.RBOUND, mu=action, sigma=self.delta, size=self.nb_actions
            )

        return np.clip(action, self.LBOUND, self.RBOUND)