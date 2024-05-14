import numpy as np
import torch

from model import AllConvModelTorch, Autoencoder
from processes import classify

if __name__ == "__main__":

    MODEL_PATH = "../checkpoints/blur/final_checkpoint-1"
    model_1 = Autoencoder()
    model_2 = AllConvModelTorch(num_classes=10,
                            num_filters=64,
                            input_shape=[3, 32, 32])
    model_2.load_state_dict(
        torch.load((MODEL_PATH) + ".torchmodel"))
    
    classify(np.random.rand(1, 3, 32, 32), model_1, model_2)