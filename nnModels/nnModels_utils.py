import torch
import torch.nn as nn
import torch.nn.init as init

def get_layers(model: torch.nn.Module):
    children = list(model.children())
    return [model] if len(children) == 0 else [ci for c in children for ci in get_layers(c)]


def freeze(layers_to_freeze):
    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False


def unfreeze(layers_to_freeze):
    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
