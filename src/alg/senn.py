import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import init
from typing import Union, Callable, List
import transformers

def filter_inner_params(params: List[nn.Parameter]):
    return [p for p in params if not hasattr(p, "__conditioner__")]


class ConditionalLinearWrapper(nn.Module):
    @staticmethod
    def wrap_model(
        model: nn.Module,
        n_hidden: Union[int, Callable],
        dim: int,
        predicate: None,
        ortho: bool = False,
        modules = None
    ):
        def _recursive_apply(module: nn.Module):
            n_wrapped = 0
            for idx, (name, mod) in enumerate(module.named_children()):
                if isinstance(mod, transformers.models.gpt2.modeling_gpt2.Conv1D):
                    setattr(module, name, ConditionalLinearWrapper(w=mod.weight.t(), b=mod.bias))
                    n_wrapped += 1
                # elif isinstance(mod, nn.Linear):
                #     setattr(module, name, ConditionalLinearWrapper(w=mod.weight, b=mod.bias))
                #     n_wrapped += 1
                else:
                    n_wrapped += _recursive_apply(mod)

            return n_wrapped

        # Recursively replace each nn.Linear in the model with a ConditionerLinear
        n_wrapped = _recursive_apply(model if not modules else modules)
        print(f"Wrapped {n_wrapped} nn.Linear modules (ignoring predicate)")

        # A convenience function to enable/disable editing
        def _set_editing(self, active: bool):
            for mod in self.modules():
                if hasattr(mod, "set_active"):
                    mod.set_active(active)

        # The parameters used for conditioning/editing ONLY (these are not adapted)
        def _phi(self):
            return [p for p in self.parameters() if hasattr(p, "__conditioner__")]
        def _nphi(self):
            return [(n,p) for (n,p) in self.named_parameters() if hasattr(p, "__conditioner__")]

        # The "knowledge" or "task" parameters
        def _theta(self):
            return [p for p in self.parameters() if not hasattr(p, "__conditioner__")]

        # New methods we'll add to the model's class
        type(model).theta = _theta
        type(model).phi = _phi
        type(model).named_phi = _nphi
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
        
    
    """ A modification on the default Linear class with a learnable gradient conditioning matrix.

    Normal linear layers compute y = Wx in the forward pass and dL/dx = W^TdL/dy in the backward
    pass. The conditioned model simply learns a separate matrix P (called `back_weight`), 
    initialized to be the same as W, so the forward pass is identical, but the backward pass is 
    dL/dx = P^TdL/dy. 
    """
    def __init__(self, n_in=None, n_out=None, w=None, b=None, mode="ip"):
        super().__init__()

        if b is None:
            if n_out is not None:
                self.bias = nn.Parameter(torch.empty(n_out))
                self.reset_bias()
            else:
                self.bias = None
        else:
            self.bias = nn.Parameter(b.data.clone())

        if w is None:
            self.weight = nn.Parameter(torch.empty(n_out, n_in))
            self.reset_weight()

            if mode == "ip":
                self.back_weight = nn.Parameter(torch.empty(n_out, n_in))
            else:
                self.back_weight = nn.Parameter(torch.empty(n_out, n_out))
            self.reset_back_weight()
        else:
            self.weight = nn.Parameter(w.data.clone())
            if mode == "ip":
                self.back_weight = nn.Parameter(w.data.clone())
            else:
                self.back_weight = nn.Parameter(torch.eye(w.shape[0], device=w.device, dtype=w.dtype))

        self.back_weight.__conditioner__ = True # An attribute so we can check if a module is a conditioner or not
        def __deepcopy__(self, memo=None):
            new_data = torch.empty_like(self.data)
            new_data[:] = self.data.clone()
            new_param = nn.Parameter(new_data)
            if hasattr(self, "__conditioner__"):
                new_param.__conditioner__ = True
            new_param.__deepcopy__ = __deepcopy__.__get__(new_param)
            return new_param

        self.back_weight.__deepcopy__ = __deepcopy__.__get__(self.back_weight)

        self.mode = mode
        self.condition = False

    def set_active(self, cond):
        self.condition = cond

    def forward(self, x):
        out = x
        if not isinstance(out, torch.Tensor):
            x = out[0]
        else:
            x = out

        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = ConditionedLinearFunction.apply(
            x, self.weight, self.bias, self.back_weight, self.condition, self.mode
        )

        if not isinstance(out, torch.Tensor):
            out = (x,) + out[1:]
        else:
            out = x

        return out

    def reset_back_weight(self):
        init.kaiming_uniform_(self.back_weight, mode="fan_out", nonlinearity="relu")

    def reset_weight(self):
        init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")

    def reset_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in ** 0.5
            init.uniform_(self.bias, -bound, bound)


# Inherit from Function
class ConditionedLinearFunction(Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, bw, condition, mode):
        ctx.condition = condition
        ctx.mode = mode
        ctx.save_for_backward(input_, weight, bias, bw)
        output = input_.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        condition, mode = ctx.condition, ctx.mode
        input_, weight, bias, bw = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_bw = None

        if mode == "sq" and condition:
            grad_output = grad_output.matmul(bw)

        if ctx.needs_input_grad[0]:
            if mode == "ip":
                grad_input = grad_output.matmul(bw if condition else weight)
            else:
                grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-1, -2).matmul(input_)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1,grad_output.shape[-1]).sum(0)
        
        return grad_input, grad_weight, grad_bias, None, None, None

def test_conditioned_linear():
    torch.manual_seed(0)
    l = nn.Linear(10, 10)
    torch.manual_seed(0)
    cl = ConditionedLinear(w=l.weight, b=l.bias)
    x = torch.randn(10, requires_grad=True)

    print("Checking output...", end="")
    assert (l(x) == cl(x)).all()
    print("pass")

    l(x).tanh().mean().backward()
    lxg = x.grad.clone()
    x.grad = None
    cl(x).tanh().mean().backward()
    clxg = x.grad.clone()
    x.grad = None
    print("Checking UNCONDITIONED gradients are the same...", end="")
    assert torch.allclose(lxg, clxg)
    print("pass")

    l(x).tanh().mean().backward()
    lxg = x.grad.clone()
    x.grad = None
    cl.set_condition(True)
    cl(x).tanh().mean().backward()
    clxg = x.grad.clone()
    x.grad = None
    cl.set_condition(False)
    print("Checking CONDITIONED gradients are different...", end="")
    assert not torch.allclose(lxg, clxg)
    print("pass")

    print("Checking no first-order conditioner parameter gradient...", end="")
    assert cl.back_weight.grad is None
    print("pass")

    print("Checking second-order conditioner gradient non-zero...", end="")
    cl.set_condition(True)
    cl(x).tanh().mean().backward(create_graph=True)
    cl.set_condition(False)
    x.grad.mean().backward()
    assert not torch.allclose(torch.zeros_like(cl.back_weight.grad), cl.back_weight.grad)
    print("pass")


if __name__ == "__main__":
    test_conditioned_linear()
