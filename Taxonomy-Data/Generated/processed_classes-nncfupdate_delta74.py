class DDPG:
    def __init__(self, nb_states, nb_actions, iter_number: int = None, hparam_override: dict = None):
        self.delta_decay = None  # Initialize the delta decay factor

    def update_delta_decay_factor(self, num_train_episode):
        # Ensure the number of training episodes is positive
        assert num_train_episode > 0, "Number of training episodes must be greater than zero."

        # Update the delta decay factor based on the number of training episodes
        if num_train_episode < 1000:
            # Specific calibrated value for episodes below 1000
            self.delta_decay = 0.9
        elif 1000 <= num_train_episode <= 3000:
            # Linear interpolation for episodes between 1000 and 3000
            # Assuming a linear decay from 0.9 to 0.5 over this range
            self.delta_decay = 0.9 - (0.4 * (num_train_episode - 1000) / 2000)
        else:
            # Constant decay factor for episodes beyond 3000
            self.delta_decay = 0.5

