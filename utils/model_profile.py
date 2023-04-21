import torch
from thop import profile, clever_format
# from thop.vision.calc_func import *
import torch.nn as nn


def M_params(model:nn.Module):
    if model == None:
        return 0
    return sum([p.numel() for p in model.parameters()])/1e6


def make_profile(model_maker, inputs, inputs_description:str)->str:
    model = model_maker()
    # print(model.device)
    flops, params = profile(model, inputs=inputs)
    flops, params = clever_format([flops,params],'%.3f')

    ans = f'inputs: {inputs_description} flops: {flops} params: {params}\nmodel:\n{repr(model)}\n'
    param_spec = ''
    for name, module in model.named_modules():
        if name.count('.')<=1:
            param_spec += f'{name}: {M_params(module)} M params\n'
    ans += param_spec
    return ans


