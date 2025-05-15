def device() -> torch.device:
    """Returns current device according to current distributed configuration.

    - `torch.device("cpu")` if no distributed configuration or torch native gloo distributed configuration
    - `torch.device("cuda:local_rank")` if torch native nccl or horovod distributed configuration
    - `torch.device("xla:index")` if XLA distributed configuration

    Returns:
        torch.device

    .. versionchanged:: 0.4.2
        Added Horovod distributed framework.
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.device()