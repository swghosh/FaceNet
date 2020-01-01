from tensorflow.keras.layers import Layer, Lambda

from tensorflow.keras import backend as K
from tensorflow import nn

from tensorflow.python.keras.layers import pooling

class L2Pooling(pooling.Pooling2D):
    def l2_pooling(self, x, ksize, strides, padding, data_format):
        x = x ** 2 # squared
        x = nn.avg_pool(x, ksize, strides, padding, 
                          data_format)
        x = x * (self.pool_size[0] * self.pool_size[1]) # squared sum
        x = x ** 0.5 # sqrt
        return x

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, name=None, **kwargs):
        super().__init__(self.l2_pooling, pool_size, strides, 
                         padding, data_format, name, *kwargs)

class LRN(Layer):
    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super().__init__(**kwargs)
    def call(self, x, mask=None):
        x = nn.local_response_normalization(x, self.n, self.k, self.alpha, self.beta)
        return x
    def get_config(self):
        config = {
            "alpha": self.alpha,
            "k": self.k,
            "beta": self.beta,
            "n": self.n
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class L2Normalize(Lambda):
    def __init__(self, **kwargs):
        super().__init__(K.l2_normalize, **kwargs)

