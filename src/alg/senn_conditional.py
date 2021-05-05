import torch
import torch.nn as nn
from typing import Callable, List

def filter_inner_params(params: List[nn.Parameter]):
    return [p for p in params if not hasattr(p, "__conditioner__")]


class ConditionalLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.eye(dim))
        self.weight.__conditioner__ = True

    def set_active(self, active: bool):
        self.active = active

    def forward(self, x):
        if self.active:
            return x @ self.weight
        else:
            return x


class ConditionalLinearWrapper(nn.Module):
    @staticmethod
    def wrap_model(
        model: nn.Module, n_hidden: int, dim: int, predicate: Callable[[nn.Module], bool]
    ):
        def _recursive_apply(module: nn.Module):
            n_wrapped = 0
            for idx, (name, mod) in enumerate(module.named_children()):
                if predicate(mod):
                    setattr(module, name, ConditionalLinearWrapper(mod, n_hidden, dim))
                    n_wrapped += 1
                else:
                    n_wrapped += _recursive_apply(mod)

            return n_wrapped

        # Recursively replace each nn.Linear in the model with a ConditionerLinear
        n_wrapped = _recursive_apply(model)
        print(f"Wrapped {n_wrapped} modules for predicate {predicate}")

        # A convenience function to enable/disable editing
        def _set_editing(self, active: bool):
            for mod in self.modules():
                if hasattr(mod, "set_active"):
                    mod.set_active(active)

        # The parameters used for conditioning/editing ONLY (these are not adapted)
        def _phi(self):
            return [p for p in self.parameters() if hasattr(p, "__conditioner__")]

        # The "knowledge" or "task" parameters
        def _theta(self):
            return [p for p in self.parameters() if not hasattr(p, "__conditioner__")]

        # New methods we'll add to the model's class
        type(model).theta = _theta
        type(model).phi = _phi
        type(model).set_editing = _set_editing

        # If this model already has `inner_params` defined, hot-swap this function to
        #  ensure it doesn't return any conditioning parameters
        if hasattr(model, "inner_params"):
            def _constructor(old_inner_params):
                def _senn_inner_params(self):
                    return filter_inner_params(old_inner_params())
                return _senn_inner_params
            _inner_params = _constructor(model.inner_params)
        else:
            def _inner_params(self):
                return self.theta()

        # Add this specifically to this instance
        model.inner_params = _inner_params.__get__(model)
        
        return model

    def __init__(self, module: nn.Module, size: int, dim: int = -1):
        super().__init__()

        self.wrapped = module
        self.weight = nn.Parameter(torch.eye(size))
        self.weight.__conditioner__ = True
        self.active = False
        self.dim = dim

    def set_active(self, active: bool):
        self.active = active

    def forward(self, x):
        out = self.wrapped(x)
        if self.active:
            out = (
                (out.movedim(self.dim, -1) @ self.weight)
                .movedim(-1, self.dim)
                .contiguous()
            )

        return out
