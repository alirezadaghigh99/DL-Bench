    def __init__(self,
                 model: torch.nn.Module,
                 loss: Union[Loss, LossFn],
                 output_types: Optional[List[str]] = None,
                 batch_size: int = 100,
                 model_dir: Optional[str] = None,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 optimizer: Optional[Optimizer] = None,
                 tensorboard: bool = False,
                 wandb: bool = False,
                 log_frequency: int = 100,
                 device: Optional[torch.device] = None,
                 regularization_loss: Optional[Callable] = None,
                 wandb_logger: Optional[WandbLogger] = None,
                 **kwargs) -> None:
        """Create a new TorchModel.

        Parameters
        ----------
        model: torch.nn.Module
            the PyTorch model implementing the calculation
        loss: dc.models.losses.Loss or function
            a Loss or function defining how to compute the training loss for each
            batch, as described above
        output_types: list of strings, optional (default None)
            the type of each output from the model, as described above
        batch_size: int, optional (default 100)
            default batch size for training and evaluating
        model_dir: str, optional (default None)
            the directory on disk where the model will be stored.  If this is None,
            a temporary directory is created.
        learning_rate: float or LearningRateSchedule, optional (default 0.001)
            the learning rate to use for fitting.  If optimizer is specified, this is
            ignored.
        optimizer: Optimizer, optional (default None)
            the optimizer to use for fitting.  If this is specified, learning_rate is
            ignored.
        tensorboard: bool, optional (default False)
            whether to log progress to TensorBoard during training
        wandb: bool, optional (default False)
            whether to log progress to Weights & Biases during training
        log_frequency: int, optional (default 100)
            The frequency at which to log data. Data is logged using
            `logging` by default. If `tensorboard` is set, data is also
            logged to TensorBoard. If `wandb` is set, data is also logged
            to Weights & Biases. Logging happens at global steps. Roughly,
            a global step corresponds to one batch of training. If you'd
            like a printout every 10 batch steps, you'd set
            `log_frequency=10` for example.
        device: torch.device, optional (default None)
            the device on which to run computations.  If None, a device is
            chosen automatically.
        regularization_loss: Callable, optional
            a function that takes no arguments, and returns an extra contribution to add
            to the loss function
        wandb_logger: WandbLogger
            the Weights & Biases logger object used to log data and metrics
        """
        super(TorchModel, self).__init__(model=model,
                                         model_dir=model_dir,
                                         **kwargs)
        self.loss = loss  # not used
        self.learning_rate = learning_rate
        self.output_types = output_types  # not used
        if isinstance(loss, Loss):
            self._loss_fn: LossFn = _StandardLoss(self, loss)
        else:
            self._loss_fn = loss
        self.batch_size = batch_size
        if optimizer is None:
            self.optimizer: Optimizer = Adam(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.regularization_loss = regularization_loss

        # Select a device.

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device
        self.model = model.to(device)

        # W&B logging
        if wandb:
            logger.warning(
                "`wandb` argument is deprecated. Please use `wandb_logger` instead. "
                "This argument will be removed in a future release of DeepChem."
            )
        if wandb and not _has_wandb:
            logger.warning(
                "You set wandb to True but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        self.wandb = wandb and _has_wandb

        self.wandb_logger = wandb_logger
        # If `wandb=True` and no logger is provided, initialize default logger
        if self.wandb and (self.wandb_logger is None):
            self.wandb_logger = WandbLogger()

        # Setup and initialize W&B logging
        if (self.wandb_logger
                is not None) and (not self.wandb_logger.initialized):
            self.wandb_logger.setup()

        # Update config with KerasModel params
        wandb_logger_config = dict(loss=loss,
                                   output_types=output_types,
                                   batch_size=batch_size,
                                   model_dir=model_dir,
                                   learning_rate=learning_rate,
                                   optimizer=optimizer,
                                   tensorboard=tensorboard,
                                   log_frequency=log_frequency,
                                   regularization_loss=regularization_loss)
        wandb_logger_config.update(**kwargs)

        if self.wandb_logger is not None:
            self.wandb_logger.update_config(wandb_logger_config)

        self.log_frequency = log_frequency
        if self.tensorboard and not _has_tensorboard:
            raise ImportError(
                "This class requires tensorboard to be installed.")
        if self.tensorboard:
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(
                self.model_dir)
        if output_types is None:
            self._prediction_outputs = None
            self._loss_outputs = None
            self._variance_outputs = None
            self._other_outputs = None
        else:
            self._prediction_outputs = []
            self._loss_outputs = []
            self._variance_outputs = []
            self._other_outputs = []
            for i, type in enumerate(output_types):
                if type == 'prediction':
                    self._prediction_outputs.append(i)
                elif type == 'loss':
                    self._loss_outputs.append(i)
                elif type == 'variance':
                    self._variance_outputs.append(i)
                else:
                    self._other_outputs.append(i)
            if len(self._loss_outputs) == 0:
                self._loss_outputs = self._prediction_outputs
        self._built = False
        self._output_functions: Dict[Any, Any] = {}
        self._optimizer_for_vars: Dict[Any, Any] = {}