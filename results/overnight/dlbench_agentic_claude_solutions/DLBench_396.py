from __future__ import annotations
import torch
from laplace.baselaplace import BaseLaplace, ParametricLaplace
from laplace.utils.enums import HessianStructure, Likelihood, SubsetOfWeights

def Laplace(model: torch.nn.Module, likelihood: Likelihood | str, subset_of_weights: SubsetOfWeights | str=SubsetOfWeights.LAST_LAYER, hessian_structure: HessianStructure | str=HessianStructure.KRON, *args, **kwargs) -> BaseLaplace:
    """Generate a Python function called Laplace that simplifies Laplace access using strings instead of different classes. The function takes in a torch.nn.Module called model, a Likelihood or string ('classification' or 'regression') called likelihood, a SubsetOfWeights or string ('last_layer', 'subnetwork', 'all') called subset_of_weights with a default value of SubsetOfWeights.LAST_LAYER, and a HessianStructure or string ('diag', 'kron', 'full', 'lowrank') called hessian_structure with a default value of HessianStructure.KRON. 

The function returns a ParametricLaplace object. If subset_of_weights is "subnetwork" and hessian_structure is not "full" or "diag", a ValueError is raised. The function then creates a dictionary mapping subclass keys to subclasses of ParametricLaplace, instantiates the chosen subclass with additional arguments, and returns the instantiated subclass.if subset_of_weights == "subnetwork" and hessian_structure not in ["full", "diag"]:
        raise ValueError(
            "Subnetwork Laplace requires a full or diagonal Hessian approximation!"
        )"""
    if subset_of_weights == "subnetwork" and hessian_structure not in ["full", "diag"]:
        raise ValueError(
            "Subnetwork Laplace requires a full or diagonal Hessian approximation!"
        )

    laplace_map = {
        subclass._key: subclass
        for subclass in _all_subclasses(ParametricLaplace)
        if hasattr(subclass, "_key")
    }
    laplace_class = laplace_map[(subset_of_weights, hessian_structure)]

    return laplace_class(model, likelihood, *args, **kwargs)

def _all_subclasses(cls) -> set:
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in _all_subclasses(c)])
