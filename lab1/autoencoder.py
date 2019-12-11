import numpy as np

class Autoencoder():
    def __init__(self,
                 input_size=256,
                 crop_size=16,
                 z=0.5,
                 lr=1e-3,
                 use_adapt_lr=False,
                 use_norm=False,
                 phase='train'):
        self.input_layers = crop_size*crop_size
        self.mid_layers = int(z*self.input_layers)
        self.input_size=input_size
        self.crop_size=crop_size
        if self.input_size % self.crop_size != 0:
            raise ValueError("incorrect input data")
        self.initializer = initializer_glorot_uniform(self.input_layers, self.mid_layers)
        self.phase=phase
        self.lr = lr
        self.use_adapt_lr = use_adapt_lr
        self.use_norm = use_norm
        self.loss = lambda x, y: ((x - y) ** 2)
        self.build()
    
    def build(self):
        self.W1 = self.initializer(size=[self.input_layers, self.mid_layers])
        self.W2 = self.initializer(size=[self.mid_layers, self.input_layers])
    
    def __call__(self, inp):
        err = []
        results = []
        size = self.input_size
        crop_size = self.crop_size
        parts = inp.reshape([size, size//crop_size, crop_size]).transpose(1, 0, 2) \
               .reshape((size//crop_size)**2, crop_size, crop_size)
        for part in parts:
            inp_part = np.expand_dims(part.flatten(), 0)
            mid, res = self.forward(inp_part)
            results.append(res.flatten().reshape(crop_size, crop_size))
            if self.phase == 'train':
                diff = res-inp_part
                err.append((diff*diff).sum())
                self.backward(inp_part, mid, diff)
        if self.phase == 'train':
            return np.sum(err)
        else:
            return np.array(results).reshape(size//crop_size, size, crop_size).transpose(1,0,2).reshape(size, size)
        
    
    def forward(self, inp):
        mid = self.encode(inp)
        return mid, self.decode(mid)
    
    def backward(self, inp, mid, err):
        lr = 1/np.dot(inp, inp.T)**2 if self.use_adapt_lr else self.lr
        self.W1 -= lr * np.dot(np.dot(inp.T, err), self.W2.T)
        
        lr = 1/np.dot(mid, mid.T)**2 if self.use_adapt_lr else self.lr
        self.W2 -= lr * np.dot(mid.T, err)
        
        if self.use_norm:
            self.W2 /= np.linalg.norm(self.W2, axis=0, keepdims=True)
            self.W1 /= np.linalg.norm(self.W1, axis=1, keepdims=True)
                          
    def encode(self, inp):
        return np.dot(inp, self.W1)
    
    def decode(self, mid):
        return np.dot(mid, self.W2)
    
    def get_weights(self):
        return [self.W1.copy(), self.W2.copy()]
    
    def set_weights(self, weights):
        self.W1, self.W2 = weights
    
    def eval(self):
        self.phase = 'test'


def initializer_glorot_uniform(input_layers, output_layers):
    limit = np.sqrt(6 / (input_layers + output_layers))
    return partial(np.random.uniform, low=-limit, high=limit)