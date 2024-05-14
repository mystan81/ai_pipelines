import math
import torch
from torch.nn import Module as TorchModule


class GlobalAveragePool_3(torch.nn.Module):
    """Global average pooling operation."""

    def forward(self, x):
        return torch.mean(x, axis=[2, 3])


class AllConvModelTorch_3(TorchModule):
    """All convolutional network architecture."""

    def __init__(self, num_classes, num_filters, input_shape, activation=torch.nn.LeakyReLU(0.2)):
        super().__init__()
        conv_args = dict(
            kernel_size=3,
            padding=(1,1))

        self.layers = torch.nn.ModuleList([])
        prev = input_shape[0]
        log_resolution = int(round(
            math.log(input_shape[1]) / math.log(2)))
        for scale in range(log_resolution - 2):
            self.layers.append(torch.nn.Conv2d(prev, num_filters << scale, **conv_args))
            self.layers.append(activation)
            prev = num_filters << (scale + 1)
            self.layers.append(torch.nn.Conv2d(num_filters << scale, prev, **conv_args))
            self.layers.append(activation)
            self.layers.append(torch.nn.AvgPool2d((2, 2)))
        self.layers.append(torch.nn.Conv2d(prev, num_classes, kernel_size=3, padding=(1,1)))
        self.layers.append(GlobalAveragePool_3())
        self.layers.append(torch.nn.Softmax(dim=1))

    def __call__(self, x, training=False):
        del training  # ignore training argument since don't have batch norm
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
        return x  