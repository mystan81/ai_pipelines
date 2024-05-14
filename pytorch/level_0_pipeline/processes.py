import numpy as np
import utils
    
def blur(x):
    x_pad = np.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)])
    x_pad = (x_pad[:, :, 1:] + x_pad[:, :, :-1])/2
    x_pad = (x_pad[:, :, :, 1:] + x_pad[:, :, :, :-1])/2
    return x_pad

def classify(x, model):
    x_pad = blur(x)
    return utils.to_numpy(model(x_pad))
    