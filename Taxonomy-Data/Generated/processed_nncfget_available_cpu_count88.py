import os
import multiprocessing
import psutil

def get_available_cpu_count(logical=True):
    try:
        if logical:
            # Return the number of logical CPUs
            return psutil.cpu_count(logical=True)
        else:
            # Return the number of physical CPU cores
            return psutil.cpu_count(logical=False)
    except Exception:
        try:
            # Fallback to using os and multiprocessing if psutil is not available
            if logical:
                return os.cpu_count() or 1
            else:
                return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else multiprocessing.cpu_count()
        except Exception:
            # If all else fails, return 1
            return 1

