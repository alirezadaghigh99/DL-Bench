import os
import torch

class DDPG:
    def __init__(self, nb_states, nb_actions, iter_number: int = None, hparam_override: dict = None):
        self.actor = None  # actor network
        self.critic = None  # critic network
        # Initialize your actor and critic networks here
        pass

    def save_model(self, output):
        # Ensure the output directory exists
        os.makedirs(output, exist_ok=True)
        
        # Define the file paths
        actor_path = os.path.join(output, 'actor.pkl')
        critic_path = os.path.join(output, 'critic.pkl')
        
        # Save the state dictionaries of the actor and critic
        if self.actor is not None:
            torch.save(self.actor.state_dict(), actor_path)
            print(f"Actor model saved to {actor_path}")
        else:
            print("Actor model is not initialized.")
        
        if self.critic is not None:
            torch.save(self.critic.state_dict(), critic_path)
            print(f"Critic model saved to {critic_path}")
        else:
            print("Critic model is not initialized.")

