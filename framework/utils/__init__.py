import numpy as np
import tensorflow as tf
import random as python_random

def reset_random_seed():
    np.random.seed(123)
    python_random.seed(1234)
    tf.random.set_seed(1234)
    