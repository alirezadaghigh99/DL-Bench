    def _loss(self, model, guide, args, kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float, float

        Computes the re-weighted wake-sleep estimators for the model (wake-theta) and the
          guide (insomnia * wake-phi + (1 - insomnia) * sleep-phi).
        Performs backward as appropriate on both, over the specified number of particles.
        """

        wake_theta_loss = torch.tensor(100.0)
        if self.model_has_params or self.insomnia > 0.0:
            # compute quantities for wake theta and wake phi
            log_joints = []
            log_qs = []

            for model_trace, guide_trace in self._get_traces(
                model, guide, args, kwargs
            ):
                log_joint = 0.0
                log_q = 0.0

                for _, site in model_trace.nodes.items():
                    if site["type"] == "sample":
                        if self.vectorize_particles:
                            log_p_site = (
                                site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                            )
                        else:
                            log_p_site = site["log_prob_sum"]
                        log_joint = log_joint + log_p_site

                for _, site in guide_trace.nodes.items():
                    if site["type"] == "sample":
                        if self.vectorize_particles:
                            log_q_site = (
                                site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                            )
                        else:
                            log_q_site = site["log_prob_sum"]
                        log_q = log_q + log_q_site

                log_joints.append(log_joint)
                log_qs.append(log_q)

            log_joints = (
                log_joints[0] if self.vectorize_particles else torch.stack(log_joints)
            )
            log_qs = log_qs[0] if self.vectorize_particles else torch.stack(log_qs)
            log_weights = log_joints - log_qs.detach()

            # compute wake theta loss
            log_sum_weight = torch.logsumexp(log_weights, dim=0)
            wake_theta_loss = -(log_sum_weight - math.log(self.num_particles)).sum()
            warn_if_nan(wake_theta_loss, "wake theta loss")

        if self.insomnia > 0:
            # compute wake phi loss
            normalised_weights = (log_weights - log_sum_weight).exp().detach()
            wake_phi_loss = -(normalised_weights * log_qs).sum()
            warn_if_nan(wake_phi_loss, "wake phi loss")

        if self.insomnia < 1:
            # compute sleep phi loss
            _model = pyro.poutine.uncondition(model)
            _guide = guide
            _log_q = 0.0

            if self.vectorize_particles:
                if self.max_plate_nesting == float("inf"):
                    self._guess_max_plate_nesting(_model, _guide, args, kwargs)
                _model = self._vectorized_num_sleep_particles(_model)
                _guide = self._vectorized_num_sleep_particles(guide)

            for _ in range(1 if self.vectorize_particles else self.num_sleep_particles):
                _model_trace = poutine.trace(_model).get_trace(*args, **kwargs)
                _model_trace.detach_()
                _guide_trace = self._get_matched_trace(
                    _model_trace, _guide, args, kwargs
                )
                _log_q += _guide_trace.log_prob_sum()

            sleep_phi_loss = -_log_q / self.num_sleep_particles
            warn_if_nan(sleep_phi_loss, "sleep phi loss")

        # compute phi loss
        phi_loss = (
            sleep_phi_loss
            if self.insomnia == 0
            else (
                wake_phi_loss
                if self.insomnia == 1
                else self.insomnia * wake_phi_loss
                + (1.0 - self.insomnia) * sleep_phi_loss
            )
        )

        return wake_theta_loss, phi_loss