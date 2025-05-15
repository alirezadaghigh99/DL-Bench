    def save_model(self, output):
        torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
        torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))