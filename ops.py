import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable


def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size if grad_bias is not None else bias
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias_result = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                if grad_bias_result is not None:
                    grad_bias = Variable(grad_bias_result.data, requires_grad=False)
                    bias_adapt = bias - grad_bias * meta_step_size
                else:
                    bias_adapt = bias
            else:
                bias_adapt = bias

        weight_adapt = weight - grad_weight * meta_step_size if grad_weight is not None else weight
        return F.linear(inputs, weight_adapt, bias_adapt)
    else:
        return F.linear(inputs, weight, bias)


def conv2d(inputs, weight, bias, meta_step_size=0.001, stride=1, padding=0, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size if grad_bias is not None else bias
            else:
                bias_adapt = bias

        else:
            grad_weight_result = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]
            grad_weight = Variable(grad_weight_result.data, requires_grad=False) if grad_weight_result is not None else None
            if bias is not None:
                grad_bias_result = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                if grad_bias_result is not None:
                    grad_bias = Variable(grad_bias_result.data, requires_grad=False)
                    bias_adapt = bias - grad_bias * meta_step_size
                else:
                    bias_adapt = bias
            else:
                bias_adapt = bias

        weight_adapt = weight - grad_weight * meta_step_size if grad_weight is not None else weight
        return F.conv2d(inputs, weight_adapt, bias_adapt, stride, padding, dilation, groups)
    else:
        return F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)


def relu(inputs):
    return F.threshold(inputs, 0, 0, inplace=True)


def maxpool(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)
