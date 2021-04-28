import torch
import torch.nn as nn
from torch.autograd import Function, gradcheck
from torch.nn import init
import math


class ConditionedLinear(nn.Module):
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
            self.back_weight.data += 1e-4 * torch.randn_like(self.back_weight.data)

        self.back_weight.__conditioner__ = True

        self.mode = mode
        self.condition = False

    def set_condition(self, cond):
        self.condition = cond

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return ConditionedLinearFunction.apply(
            x, self.weight, self.bias, self.back_weight, self.condition, self.mode
        )

    def reset_back_weight(self):
        init.kaiming_uniform_(self.back_weight, mode="fan_out", nonlinearity="relu")

    def reset_weight(self):
        init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")

    def reset_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
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
