from distutils.version import LooseVersion
import logging
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
import numpy as np
import warnings
import time

multiply_adds = 1

def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

def l_sum(in_list):
    res = 0
    for _ in in_list:
        res += _
    return res

def calculate_parameters(param_list):
    total_params = 0
    for p in param_list:
        total_params += torch.DoubleTensor([p.nelement()])
    return total_params

def calculate_zero_ops():
    return torch.DoubleTensor([int(0)])

def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])

def calculate_conv(bias, kernel_size, output_size, in_channel, group):
    warnings.warn("This API is being deprecated.")
    return torch.DoubleTensor([output_size * (in_channel / group * kernel_size + bias)])

def calculate_norm(input_size):
    return torch.DoubleTensor([2 * input_size])

def calculate_relu_flops(input_size):
    return 0
    
def calculate_relu(input_size: torch.Tensor):
    warnings.warn("This API is being deprecated")
    return torch.DoubleTensor([int(input_size)])

def calculate_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])

def calculate_avgpool(input_size):
    return torch.DoubleTensor([int(input_size)])

def calculate_adaptive_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])

def calculate_upsample(mode: str, output_size):
    total_ops = output_size
    if mode == "linear":
        total_ops *= 5
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "bicubic":
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops *= ops_solve_A + ops_solve_p
    elif mode == "trilinear":
        total_ops *= 13 * 2 + 5
    return torch.DoubleTensor([int(total_ops)])

def calculate_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])

def counter_matmul(input_size, output_size):
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(input_size) * output_size[-1]

def counter_mul(input_size):
    return input_size

def counter_pow(input_size):
    return input_size

def counter_sqrt(input_size):
    return input_size

def counter_div(input_size):
    return input_size

def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    m.total_params[0] = calculate_parameters(m.parameters())

def zero_ops(m, x, y):
    m.total_ops += calculate_zero_ops()

def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    x = x[0]
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.weight.shape),
        groups = m.groups,
        bias = m.bias
    )

def count_convNd_ver2(m: _ConvNd, x, y: torch.Tensor):
    x = x[0]
    output_size = torch.zeros((y.size()[:1] + y.size()[2:])).numel()
    m.total_ops += calculate_conv(m.bias.nelement(), m.weight.nelement(), output_size)

def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    x = x[0]
    flops = calculate_norm(x.numel())
    if (getattr(m, 'affine', False) or getattr(m, 'elementwise_affine', False)):
        flops *= 2
    m.total_ops += flops

def count_prelu(m, x, y):
    x = x[0]
    nelements = x.numel()
    if not m.training:
        m.total_ops += calculate_relu(nelements)

def count_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    m.total_ops += calculate_relu_flops(list(x.shape))

def count_softmax(m, x, y):
    x = x[0]
    nfeatures = x.size()[m.dim]
    batch_size = x.numel() // nfeatures
    m.total_ops += calculate_softmax(batch_size, nfeatures)

def count_avgpool(m, x, y):
    num_elements = y.numel()
    m.total_ops += calculate_avgpool(num_elements)

def count_adap_avgpool(m, x, y):
    kernel = torch.div(
        torch.DoubleTensor([*(x[0].shape[2:])]), 
        torch.DoubleTensor([*(y.shape[2:])])
    )
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    m.total_ops += calculate_adaptive_avg(total_add, num_elements)

def count_upsample(m, x, y):
    if m.mode not in (
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
    ):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        m.total_ops += 0
    else:
        x = x[0]
        m.total_ops += calculate_upsample(m.mode, y.nelement())


def count_linear(m, x, y):
    total_mul = m.in_features
    num_elements = y.numel()
    m.total_ops += calculate_linear(total_mul, num_elements)


register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,
    nn.BatchNorm1d: count_normalization,
    nn.BatchNorm2d: count_normalization,
    nn.BatchNorm3d: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.InstanceNorm1d: count_normalization,
    nn.InstanceNorm2d: count_normalization,
    nn.InstanceNorm3d: count_normalization,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,
    nn.Sequential: zero_ops,
    nn.PixelShuffle: zero_ops,
}

def count_linear(m, x, y):
    num_elements = y.numel()
    m.total_ops += torch.DoubleTensor([int(m.in_features * num_elements)])

if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
    register_hooks.update({nn.SyncBatchNorm: count_normalization})

def getFlops_macs(
    model: nn.Module,
    inputs=torch.randn(1, 3, 224, 224),

):
    handler_collection = {}
    types_collection = set()

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m_type = type(m)

        fn = None
        if m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is not None:
            handler_collection[m] = (m.register_forward_hook(fn))

        types_collection.add(m_type)
    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops = module.total_ops.item()
        for n, m in module.named_children():
            if m in handler_collection and not isinstance(
                m, (nn.Sequential, nn.ModuleList)
            ):
                m_ops = m.total_ops.item()
            else:
                m_ops = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops

        return total_ops
    total_ops = dfs_count(model)

    model.train(prev_training_status)
    for m, op_handler in handler_collection.items():
        op_handler.remove()
        m._buffers.pop("total_ops")
    return total_ops

def throughput(data_loader, model, nAvg=30):
    model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            H, W = images.shape[2], images.shape[3]
            for i in range(50):
                model(images)
            torch.cuda.synchronize()
            tic1 = time.time()
            for i in range(nAvg):
                model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            return nAvg, batch_size, (H, W), nAvg * batch_size / (tic2 - tic1)

def total_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)