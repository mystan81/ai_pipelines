import numpy as np
import torch

from model import LSTMModel_5
from processes import crop

if __name__ == "__main__":
    model = LSTMModel_5(input_dim=64, hidden_dim=100, layer_dim=1, output_dim=10)
    MODEL_PATH = "../checkpoints/lstm/final_checkpoint-1"
    model.load_state_dict(
        torch.load((MODEL_PATH) + ".torchmodel"))
    model(crop(np.random.rand(1, 100), 64))
    

    