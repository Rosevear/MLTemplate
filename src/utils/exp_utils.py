import random
import numpy as np

def freeze_random_generators(random_seed):
    """
    Configures all of the random processed that depend on following packages,
    so that they start from the provided random_seed variable.

    This is done to ensure the reproducibility of experiments that make use of these libraries internally

    The standard library random package
    The numpy random package NOTE: The random seed state for numpy is not threadsafe, meaning it is shared across different threads run concurrently. See https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
    TODO: Tensorflow
    """

    random.seed(random_seed)
    np.random.seed(random_seed)




