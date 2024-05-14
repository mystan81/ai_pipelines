import numpy as np
import torch

from model import AllConvModelTorch_5, Autoencoder_5
from processes import classify

if __name__ == "__main__":

    MODEL_PATH = "../checkpoints/blur/final_checkpoint-1"
    model_1 = Autoencoder_5()
    model_2 = AllConvModelTorch_5(num_classes=10,
                            num_filters=64,
                            input_shape=[3, 32, 32])
    model_2.load_state_dict(
        torch.load((MODEL_PATH) + ".torchmodel"))
    
    classify(np.random.rand(1, 3, 32, 32), model_1, model_2)