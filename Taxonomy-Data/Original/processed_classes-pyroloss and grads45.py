    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float

        Computes the RWS estimators for the model (wake-theta) and the guide (wake-phi).
        Performs backward as appropriate on both, using num_particle many samples/particles.
        """
        wake_theta_loss, phi_loss = self._loss(model, guide, args, kwargs)
        # convenience addition to ensure easier gradients without requiring `retain_graph=True`
        (wake_theta_loss + phi_loss).backward()

        return wake_theta_loss.detach().item(), phi_loss.detach().item()