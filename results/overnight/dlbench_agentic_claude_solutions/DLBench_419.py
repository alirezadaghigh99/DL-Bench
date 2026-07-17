import importlib.util
import platform
import sys
import warnings
from dataclasses import asdict, fields, is_dataclass
from functools import wraps
from inspect import isclass, isfunction
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, overload
import torch
from torch.linalg import inv_ex
from kornia.core import Tensor
from kornia.utils._compat import torch_version_ge

def xla_is_available() -> bool:
    """Return whether `torch_xla` is available in the system."""
    if importlib.util.find_spec('torch_xla') is not None:
        return True
    return False

def is_mps_tensor_safe(x: Tensor) -> bool:
    """Return whether tensor is on MPS device."""
    return 'mps' in str(x.device)

def get_cuda_device_if_available(index: int=0) -> torch.device:
    """Tries to get cuda device, if fail, returns cpu.

    Args:
        index: cuda device index

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{index}')
    return torch.device('cpu')

def get_mps_device_if_available() -> torch.device:
    """Tries to get mps device, if fail, returns cpu.

    Returns:
        torch.device
    """
    dev = 'cpu'
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            dev = 'mps'
    return torch.device(dev)

def get_cuda_or_mps_device_if_available() -> torch.device:
    """Checks OS and platform and runs get_cuda_device_if_available or get_mps_device_if_available.

    Returns:
        torch.device
    """
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        return get_mps_device_if_available()
    else:
        return get_cuda_device_if_available()

@overload
def map_location_to_cpu(storage: Tensor, location: str) -> Tensor:
    ...

@overload
def map_location_to_cpu(storage: str) -> str:
    ...

def map_location_to_cpu(storage: Union[str, Tensor], *args: Any, **kwargs: Any) -> Union[str, Tensor]:
    """Map location of device to CPU, util for loading things from HUB."""
    return storage

def deprecated(replace_with: Optional[str]=None, version: Optional[str]=None, extra_reason: Optional[str]=None) -> Any:

    def _deprecated(func: Callable[..., Any]) -> Any:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = ''
            beginning = f'Since kornia {version} the ' if version is not None else ''
            if isclass(func):
                name = func.__class__.__name__
            if isfunction(func):
                name = func.__name__
            warnings.simplefilter('always', DeprecationWarning)
            if replace_with is not None:
                warnings.warn(f'{beginning}`{name}` is deprecated in favor of `{replace_with}`.{extra_reason}', category=DeprecationWarning, stacklevel=2)
            else:
                warnings.warn(f'{beginning}`{name}` is deprecated and will be removed in the future versions.{extra_reason}', category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return wrapper
    return _deprecated

def _extract_device_dtype(tensor_list: List[Optional[Any]]) -> Tuple[torch.device, torch.dtype]:
    """Check if all the input are in the same device (only if when they are Tensor).

    If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).

    Returns:
        [torch.device, torch.dtype]
    """
    (device, dtype) = (None, None)
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (Tensor,)):
                continue
            _device = tensor.device
            _dtype = tensor.dtype
            if device is None and dtype is None:
                device = _device
                dtype = _dtype
            elif device != _device or dtype != _dtype:
                raise ValueError(f'Passed values are not in the same device and dtype.Got ({device}, {dtype}) and ({_device}, {_dtype}).')
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    return (device, dtype)

def _torch_inverse_cast(input: Tensor) -> Tensor:
    """Helper function to make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, Tensor):
        raise AssertionError(f'Input must be Tensor. Got: {type(input)}.')
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.linalg.inv(input.to(dtype)).to(input.dtype)

def _torch_histc_cast(input: Tensor, bins: int, min: int, max: int) -> Tensor:
    """Helper function to make torch.histc work with other than fp32/64.

    The function torch.histc is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, Tensor):
        raise AssertionError(f'Input must be Tensor. Got: {type(input)}.')
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.histc(input.to(dtype), bins, min, max).to(input.dtype)

def _torch_svd_cast(input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Helper function to make torch.svd work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
    """
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    (out1, out2, out3H) = torch.linalg.svd(input.to(dtype))
    if torch_version_ge(1, 11):
        out3 = out3H.mH
    else:
        out3 = out3H.transpose(-1, -2)
    return (out1.to(input.dtype), out2.to(input.dtype), out3.to(input.dtype))

def _torch_linalg_svdvals(input: Tensor) -> Tensor:
    """Helper function to make torch.linalg.svdvals work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
    """
    if not isinstance(input, Tensor):
        raise AssertionError(f'Input must be Tensor. Got: {type(input)}.')
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    if TYPE_CHECKING:
        out: Tensor
    elif torch_version_ge(1, 10):
        out = torch.linalg.svdvals(input.to(dtype))
    else:
        (_, out, _) = torch.linalg.svd(input.to(dtype))
    return out.to(input.dtype)

def _torch_solve_cast(A: Tensor, B: Tensor) -> Tensor:
    """Helper function to make torch.solve work with other than fp32/64.

    For stable operation, the input matrices should be cast to fp64, and the output will be cast back to the input
    dtype.
    """
    out = torch.linalg.solve(A.to(torch.float64), B.to(torch.float64))
    return out.to(A.dtype)

def safe_solve_with_mask(B: Tensor, A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Write a python function The `safe_solve_with_mask` function is designed to solve a system of linear equations \\( AX = B \\) while handling potential issues related to singular matrices. The function provides a safe mechanism to avoid crashes due to singular matrices by returning a mask that indicates valid solutions. Here's a breakdown of the input, output, and error handling:

### Input:
- **`B` (Tensor):** The right-hand side tensor in the equation \\( AX = B \\).
- **`A` (Tensor):** The matrix tensor that will be solved against \\( B \\).

### Output:
- **`X` (Tensor):** The solution tensor that satisfies \\( AX = B \\), or a closest approximation if the matrix is near-singular.
- **`A_LU` (Tensor):** The LU decomposition of matrix `A`, which is useful for numerical stability.
- **`valid_mask` (Tensor):** A boolean tensor indicating which rows of the batch were solved successfully (i.e., without encountering a singular matrix).

### Error Handling:
- If the PyTorch version is less than 1.10, the function falls back to a different solution method (`_torch_solve_cast`) and issues a warning that the validity mask might not be correct.
- If `B` is not a tensor, an `AssertionError` is raised.
- If the data type of `B` is not `torch.float32` or `torch.float64`, it is cast to `torch.float32` for compatibility with the solution functions.
- The function includes branches for compatibility with different PyTorch versions, ensuring that it can handle LU decomposition and solving in a way that is consistent with the PyTorch version in use.

This function is useful in scenarios where solving linear equations might encounter singular matrices, and you need a robust method to handle such cases gracefully."""
    if not torch_version_ge(1, 10):
        sol = _torch_solve_cast(A, B)
        warnings.warn('PyTorch version < 1.10, solve validity mask maybe not correct', RuntimeWarning)
        return (sol, sol, torch.ones(1, dtype=torch.bool, device=B.device))
    if not isinstance(B, Tensor):
        raise AssertionError(f'B must be Tensor. Got: {type(B)}.')
    dtype: torch.dtype = B.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    if torch_version_ge(1, 13):
        (A_LU, pivots, info) = torch.linalg.lu_factor_ex(A.to(dtype))
    else:
        (A_LU, pivots, info) = torch.lu(A.to(dtype), get_infos=True, pivot=True)
    valid_mask: Tensor = info == 0
    if torch_version_ge(1, 13):
        X = torch.linalg.lu_solve(A_LU, pivots, B.to(dtype))
    else:
        X = torch.lu_solve(B.to(dtype), A_LU, pivots)
    return (X.to(B.dtype), A_LU.to(A.dtype), valid_mask)

def safe_inverse_with_mask(A: Tensor) -> Tuple[Tensor, Tensor]:
    """Helper function, which avoids crashing because of non-invertable matrix input and outputs the mask of valid
    solution."""
    if not isinstance(A, Tensor):
        raise AssertionError(f'A must be Tensor. Got: {type(A)}.')
    dtype_original = A.dtype
    if dtype_original not in (torch.float32, torch.float64):
        dtype = torch.float32
    else:
        dtype = dtype_original
    (inverse, info) = inv_ex(A.to(dtype))
    mask = info == 0
    return (inverse.to(dtype_original), mask)

def is_autocast_enabled(both: bool=True) -> bool:
    """Check if torch autocast is enabled.

    Args:
        both: if True will consider autocast region for both types of devices

    Returns:
        Return a Bool,
        will always return False for a torch without support, otherwise will be: if both is True
        `torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu')`. If both is False will return just
        `torch.is_autocast_enabled()`.
    """
    if TYPE_CHECKING:
        return False
    if not torch_version_ge(1, 10, 2):
        return False
    if both:
        if torch_version_ge(2, 4):
            return torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu')
        else:
            return torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
    return torch.is_autocast_enabled()

def dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dictionaries."""
    if is_dataclass(obj) and (not isinstance(obj, type)):
        return {key: dataclass_to_dict(value) for (key, value) in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)((dataclass_to_dict(item) for item in obj))
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for (key, value) in obj.items()}
    else:
        return obj
T = TypeVar('T')

def dict_to_dataclass(dict_obj: Dict[str, Any], dataclass_type: Type[T]) -> T:
    """Recursively convert dictionaries to dataclass instances."""
    if not isinstance(dict_obj, dict):
        raise TypeError('Input conf must be dict')
    if not is_dataclass(dataclass_type):
        raise TypeError('dataclass_type must be a dataclass')
    field_types: dict[str, Any] = {f.name: f.type for f in fields(dataclass_type)}
    constructor_args = {}
    for (key, value) in dict_obj.items():
        if key in field_types and is_dataclass(field_types[key]):
            constructor_args[key] = dict_to_dataclass(value, field_types[key])
        else:
            constructor_args[key] = value
    return dataclass_type(**constructor_args)
