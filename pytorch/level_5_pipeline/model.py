import math
import torch
from torch.nn import Module as TorchModule


class GlobalAveragePool_5(torch.nn.Module):
    """Global average pooling operation."""

    def forward(self, x):
        return torch.mean(x, axis=[2, 3])


class AllConvModelTorch_5(TorchModule):
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
        self.layers.append(GlobalAveragePool_5())
        self.layers.append(torch.nn.Softmax(dim=1))

    def __call__(self, x, training=False):
        del training  # ignore training argument since don't have batch norm
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
        return x
    
class Autoencoder_5(TorchModule):
    def __init__(self):
        super(Autoencoder_5, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),  # to ensure the output is between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class LSTMModel_5(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel_5, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out