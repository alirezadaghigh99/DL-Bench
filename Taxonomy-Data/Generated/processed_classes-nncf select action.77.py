import numpy as np
from scipy.stats import truncnorm

class DDPG:
    LBOUND = 0.0
    RBOUND = 1.0

    def __init__(self, nb_states, nb_actions, iter_number: int = None, hparam_override: dict = None):
        self.actor = None  # This should be your actor network
        self.init_delta = 0.5  # initial delta for noise
        self.delta_decay = 0.995  # decay rate for delta
        self.warmup_iter_number = 20  # number of warmup iterations
        self.nb_actions = nb_actions  # number of actions
        self.delta = self.init_delta  # current delta for noise

    def select_action(self, s_t, episode, decay_epsilon=True):
        # Predict the action using the actor network
        action = self.actor.predict(s_t)

        if decay_epsilon:
            # Decay the delta over time
            self.delta = self.init_delta * (self.delta_decay ** episode)

            # Apply noise sampled from a truncated normal distribution
            noise = truncnorm.rvs(
                (self.LBOUND - action) / self.delta, 
                (self.RBOUND - action) / self.delta, 
                loc=0, 
                scale=self.delta, 
                size=self.nb_actions
            )
            action += noise

        # Clip the action to be within the bounds
        action = np.clip(action, self.LBOUND, self.RBOUND)

        return action

# Note: The `actor` network should have a `predict` method that takes the state `s_t` and returns the action.