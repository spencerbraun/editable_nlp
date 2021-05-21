import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Union
import inspect


def filter_inner_params(params: List[nn.Parameter]):
    return [p for p in params if not hasattr(p, "__conditioner__")]


class ConditionalLinearWrapper(nn.Module):
    @staticmethod
    def wrap_model(
        model: nn.Module,
        n_hidden: Union[int, Callable],
        dim: int,
        predicate: Callable[[nn.Module], bool],
        ortho: bool = False,
        modules = None
    ):
        def _recursive_apply(module: nn.Module):
            n_wrapped = 0
            for idx, (name, mod) in enumerate(module.named_children()):
                if predicate(mod):
                    num_hidden = n_hidden(mod) if isinstance(n_hidden, Callable) else n_hidden
                    setattr(module, name, ConditionalLinearWrapper(mod, num_hidden, dim, ortho=ortho))
                    n_wrapped += 1
                else:
                    n_wrapped += _recursive_apply(mod)

            return n_wrapped

        # Recursively replace each nn.Linear in the model with a ConditionerLinear
        n_wrapped = _recursive_apply(model if not modules else modules)
        print(f"Wrapped {n_wrapped} modules using predicate:\n{inspect.getsource(predicate)}")

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
            print("Overriding existing `inner_params` implementation")
            type(model).default_inner_params = type(model).inner_params
            type(model).inner_params = lambda self: filter_inner_params(
                self.default_inner_params()
            )
        else:
            print("Injecting defaulting `inner_params` implementation -> `self.theta`")
            type(model).default_inner_params = model.theta.__func__
            type(model).inner_params = model.theta.__func__

        print(f"n default inner params: {len(model.default_inner_params())}")
        print(f"n inner params: {len(model.inner_params())}")

        return model

    def __init__(self, module: nn.Module, size: int, dim: int = -1, ortho: bool = False):
        super().__init__()

        self.wrapped = module
        if ortho:
            self.idxs = torch.triu_indices(size,size)
        else:
            self.idxs = None

        self.weight = nn.Parameter(torch.eye(size) if not ortho else torch.zeros(size, size))
        self.weight.__conditioner__ = True

        self.active = False
        self.dim = dim
        self.ortho = ortho

        def __deepcopy__(self, memo=None):
            new_data = torch.empty_like(self.data)
            new_data[:] = self.data.clone()
            new_param = nn.Parameter(new_data)
            if hasattr(self, "__conditioner__"):
                new_param.__conditioner__ = True
            new_param.__deepcopy__ = __deepcopy__.__get__(new_param)
            return new_param

        self.weight.__deepcopy__ = __deepcopy__.__get__(self.weight)

    def set_active(self, active: bool):
        self.active = active

    def forward(self, *args, **kwargs):
        out = self.wrapped(*args, **kwargs)
        if self.active:
            if not isinstance(out, torch.Tensor):
                x = out[0]
            else:
                x = out

            try:
                if self.ortho:
                    I = torch.eye(self.weight.shape[0], device=self.weight.device)
                    A = self.weight.triu(1) - self.weight.triu(1).permute(-1,-2)
                    weight = (I + A) @ (I - A).inverse()
                else:
                    weight = self.weight

                x = x.movedim(self.dim, -1)
                x = x @ weight
                x = x.movedim(-1, self.dim).contiguous()
                
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()

            if not isinstance(out, torch.Tensor):
                out = (x,) + out[1:]
            else:
                out = x

        return out


def _test_copy(model):
    import copy

    print("default", len(model.default_inner_params()))
    print("inner", len(model.inner_params()))
    modelcopy = copy.deepcopy(model)
    print("copied")
    print("default", len(modelcopy.default_inner_params()))
    print("inner", len(modelcopy.inner_params()))
    if id(modelcopy.inner_params()[0].data) != id(model.inner_params()[0].data):
        print("FAIL")
    else:
        print("SUCCESS")


if __name__ == "__main__":
    import transformers
    import utils

    model = transformers.GPT2LMHeadModel.from_pretrained("distilgpt2")
    tok = transformers.GPT2Tokenizer.from_pretrained("distilgpt2")
    utils.prep_for_maml(model)
    conv_predicate = lambda mod: (
        isinstance(mod, transformers.models.gpt2.modeling_gpt2.Conv1D)
        and mod.weight.shape[1] == 768
    )
    ConditionalLinearWrapper.wrap_model(model, model.config.n_embd, -1, conv_predicate)

    input_ids = tok("This is a test sequence", return_tensors="pt")["input_ids"]

    grads1 = torch.autograd.grad(
        model(input_ids, labels=input_ids).loss, model.phi(), allow_unused=True
    )
    model.set_editing(True)
    print("editing ON")
    grads2 = torch.autograd.grad(model(input_ids, labels=input_ids).loss, model.phi())
    model.set_editing(False)
    print("editing OFF")
    grads3 = torch.autograd.grad(
        model(input_ids, labels=input_ids).loss, model.phi(), allow_unused=True
    )

    import pdb

    pdb.set_trace()
