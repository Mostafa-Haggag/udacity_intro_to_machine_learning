import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import time
import json
import tensorflow_hub as hub
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras import layers
import argparse
from PIL import Image


batch_size = 32
image_size = 224
class_names = {}


def format_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def load_model(saved_keras_model_filepath):
    loaded_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})
    return loaded_model

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = format_image(image)
    expanded_image = np.expand_dims(image, axis=0)
    prob_list = model.predict(expanded_image)
    classes = []
    probs = []
    rank = prob_list[0].argsort()[::-1]
    for i in range(top_k):
        
        index = rank[i] + 1
        cls = class_names[str(index)]
        
        probs.append(prob_list[0][index])
        classes.append(cls)
    
    return probs, classes
    
    
if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    print(args)
    
    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    image_path = args.arg1
    
    model = load_model('my_model.h5')
    top_k = args.top_k

    if top_k is None: 
        top_k = 5

    with open('label_map.json', 'r') as f:
  	  class_names = json.load(f)   
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    print(classes)