def get_available_cpu_count(logical: bool = True) -> int:
    """
    Return the number of CPUs in the system.

    :param logical: If False return the number of physical cores only (e.g. hyper thread CPUs are excluded),
      otherwise number of logical cores. Defaults, True.
    :return: Number of CPU.
    """
    try:
        num_cpu = psutil.cpu_count(logical=logical)
        return num_cpu if num_cpu is not None else 1
    except Exception:
        return 1