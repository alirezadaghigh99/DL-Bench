import torch
import os

def device():
    # Check if distributed is initialized
    if torch.distributed.is_initialized():
        # Get the current backend
        backend = torch.distributed.get_backend()
        
        if backend == 'nccl':
            # For NCCL, we assume CUDA is available
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            return torch.device(f"cuda:{local_rank}")
        elif backend == 'gloo':
            # Gloo typically runs on CPU
            return torch.device("cpu")
    else:
        # Check for XLA environment variable
        if 'XRT_TPU_CONFIG' in os.environ:
            # Assuming XLA is being used
            index = int(os.environ.get('XLA_INDEX', 0))
            return torch.device(f"xla:{index}")
    
    # Default to CPU if no distributed configuration is found
    return torch.device("cpu")

