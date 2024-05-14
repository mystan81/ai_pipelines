import math
import numpy as np
import utils
import torch
from torch.nn import Module as TorchModule


class GlobalAveragePool_1(torch.nn.Module):
    """Global average pooling operation."""

    def forward(self, x):
        return torch.mean(x, axis=[2, 3])


class AllConvModelTorch_1(TorchModule):
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
        self.layers.append(GlobalAveragePool_1())
        self.layers.append(torch.nn.Softmax(dim=1))

    def __call__(self, x, training=False):
        del training  # ignore training argument since don't have batch norm
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
        return x
    
def blur(x):
    x_pad = np.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)])
    x_pad = (x_pad[:, :, 1:] + x_pad[:, :, :-1])/2
    x_pad = (x_pad[:, :, :, 1:] + x_pad[:, :, :, :-1])/2
    return x_pad

def classify(x, model):
    x_pad = blur(x)
    return utils.to_numpy(model(x_pad))

def run():
    MODEL_PATH = "../checkpoints/blur/final_checkpoint-1"
    model = AllConvModelTorch_1(num_classes=10,
                            num_filters=64,
                            input_shape=[3, 32, 32])
    model.load_state_dict(
        torch.load((MODEL_PATH) + ".torchmodel"))
    
    classify(np.random.rand(1, 3, 32, 32), model)

if __name__ == "__main__":
    run()

    