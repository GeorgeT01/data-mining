'''
George Alromhin gr.858301

[1]Ali Abdelaal, Autoencoders for Image Reconstruction in Python and Keras
https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/

[2]Aditya Sharma, Autoencoder as a Classifier
https://www.datacamp.com/community/tutorials/autoencoder-classifier-python

[3]Aditya Sharma,Understanding Autoencoders
https://www.learnopencv.com/understanding-autoencoders-using-tensorflow-python/
'''


import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

from os.path import join as pjoin
from functools import partial
from tqdm import tqdm
from IPython.core.display import Image, display


#To test the hypotheses, the VOC 2012 dataset was used, the image has the size of 256x256 and it's grayscale image.
#The first generator takes the image from the dataset and always returns it the same.
gen_img_path = None
def first_generator(image, c_max = 255):
    global gen_img_path
    gen_img_path = image
    img = cv2.imread(image,0) #grayscale
    img = 2*img/c_max - 1 #C i (jk) = 2*C i (jk) / C max) – 1
    while 1:
        yield img.copy()


#The second returns the image.
def second_generator(image, c_max = 255):
    while 1:
        img = image
        img = 2*image/c_max - 1
        yield img

_image = "images/house_est_s5.png"
gen1 = first_generator(_image)
gen2 = second_generator(_image)


def initializer_glorot_uniform(input_layers, output_layers):
    limit = np.sqrt(6 / (input_layers + output_layers))
    return partial(np.random.uniform, low=-limit, high=limit)



#Autoencoder
#Create an Autoencoder class that takes a 256x256 grayscale image, splits it into 16x16 cropes (or others, the user specifies) 
#and z compression coefficient, and based on this creates a learning model.
#usb_adapt_lr-Flag indicating whether to use adaptive learning rate
#use_norm-Flag indicating whether to normalize weights after they are updated
class Autoencoder():
    def __init__(self,
                 z=0.5,
                 lr=1e-3,
                 use_adapt_lr=False,
                 use_norm=False,
                 phase='train'):
        self.input_layers = 256 #crop size * crop size (16*16)
        self.mid_layers = int(z*self.input_layers)
        self.input_size = 256
        self.crop_size = 16
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

def predict(model):
    img = cv2.imread(_image, 0)
    h, w = img.shape
    res = model(img)
    fig, ax = plt.subplots(1,2)
    
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(res, cmap='gray')

def train_image(use_norm=True, use_adapt_lr=True, z=0.75):
    model = Autoencoder(use_norm=use_norm, use_adapt_lr=use_adapt_lr, z=z)
    errors = []
    it_count = 300
    best_weights = None
    best_error = np.inf
    for it in tqdm(range(it_count)):
        inp = next(gen1)
        err = model(inp)
        errors.append(err)
        if err < best_error:
            best_error = err
            best_weights = model.get_weights()
            #print(best_error)

    x = np.arange(len(errors))
    plt.xlabel("iterations")
    plt.ylabel("error")
    plt.plot(x, np.array(errors))
    idx = np.argmin(errors)
    print("BEST ERROR {}".format(errors[idx]))
    plt.plot(x[idx], errors[idx], 'rx--', linewidth=2, markersize=12)
    
    model.eval()
    model.set_weights(best_weights)
    assert gen_img_path is not None
    predict(model)
    
    plt.show()

#Check at what selection of hyperparameters will optimally converge,
#the results on the graph
for use_norm in [False, True]:
    for use_adapt_lr in [False, True]:
        train_image(use_norm, use_adapt_lr)
