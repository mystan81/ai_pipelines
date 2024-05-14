import numpy as np
import torch

from model import AllConvModelTorch_4
from processes import classify

if __name__ == "__main__":

    MODEL_PATH = "../checkpoints/blur/final_checkpoint-1"
    model = AllConvModelTorch_4(num_classes=10,
                            num_filters=64,
                            input_shape=[3, 32, 32])
    model.load_state_dict(
        torch.load((MODEL_PATH) + ".torchmodel"))
    
    classify(np.random.rand(1, 3, 32, 32), model)