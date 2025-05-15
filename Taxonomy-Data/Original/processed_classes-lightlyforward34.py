    def forward(
        self,
        high_resolution_outputs: List[torch.Tensor],
        low_resolution_outputs: List[torch.Tensor],
        queue_outputs: List[torch.Tensor] = None,
    ):
        """Computes the SwaV loss for a set of high and low resolution outputs.

        - [0]: SwaV, 2020, https://arxiv.org/abs/2006.09882

        Args:
            high_resolution_outputs:
                List of similarities of features and SwaV prototypes for the
                high resolution crops.
            low_resolution_outputs:
                List of similarities of features and SwaV prototypes for the
                low resolution crops.
            queue_outputs:
                List of similarities of features and SwaV prototypes for the
                queue of high resolution crops from previous batches.

        Returns:
            Swapping assignments between views loss (SwaV) as described in [0].
        """
        n_crops = len(high_resolution_outputs) + len(low_resolution_outputs)

        # Multi-crop iterations
        loss = 0.0
        for i in range(len(high_resolution_outputs)):
            # Compute codes of i-th high resolution crop
            with torch.no_grad():
                outputs = high_resolution_outputs[i].detach()

                # Append queue outputs
                if queue_outputs is not None:
                    outputs = torch.cat((outputs, queue_outputs[i].detach()))

                # Compute the codes
                q = sinkhorn(
                    outputs,
                    iterations=self.sinkhorn_iterations,
                    epsilon=self.sinkhorn_epsilon,
                    gather_distributed=self.sinkhorn_gather_distributed,
                )

                # Drop queue similarities
                if queue_outputs is not None:
                    q = q[: len(high_resolution_outputs[i])]

            # Compute subloss for each pair of crops
            subloss = 0.0
            for v in range(len(high_resolution_outputs)):
                if v != i:
                    subloss += self.subloss(high_resolution_outputs[v], q)

            for v in range(len(low_resolution_outputs)):
                subloss += self.subloss(low_resolution_outputs[v], q)

            loss += subloss / (n_crops - 1)

        return loss / len(high_resolution_outputs)