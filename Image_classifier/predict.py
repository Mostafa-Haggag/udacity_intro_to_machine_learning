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
Image_resolution = 224
class_names = {}


def process_np_image(img):
    processed_image = tf.convert_to_tensor(img, dtype=tf.float32)
    processed_image = tf.image.resize(processed_image, (Image_resolution, Image_resolution))
    processed_image /= 255
    return processed_image.numpy()
def predict(image_path, model, top_k):
    image = Image.open(image_path)# opens the image
    Numpy_test_image = np.asarray(image)#convert the input to array
    Numpy_test_image_proccessed = process_np_image(Numpy_test_image)# do the normal normalizing
    expanded_image = np.expand_dims(Numpy_test_image_proccessed, axis=0)# takes and array and the place of the axis that we will expand on
    # change from (224, 224) to (1, 224, 224)
    probes = model.predict(expanded_image)
    top_k_values, top_k_indices = tf.nn.top_k(probes, k=top_k)
    data = []
    for value in top_k_indices.cpu().numpy()[0]:
        data.append(class_names[str(value+1)])
    return top_k_values.numpy()[0], data
if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',default='./test_images/hard-leaved_pocket_orchid.jpg')
    parser.add_argument('--model',default='my_model_1618776856.h5')
    parser.add_argument('--top_k',default = 5)
    parser.add_argument('--category_names',default = 'label_map.json') 
    
    
    input_by_user = parser.parse_args()
    print(input_by_user)
    
    print('image_dir:', input_by_user.image_dir)
    print('model:', input_by_user.model)
    print('top_k:', input_by_user.top_k)
    print('category_names:', input_by_user.category_names)
    
    image_path = input_by_user.image_dir
    model_path = input_by_user.model
    Max_output = input_by_user.top_k
    classes    = input_by_user.category_names
    with open(classes, 'r') as f:
      class_names = json.load(f)
    if Max_output is None: 
        Max_output = 5
    model =  tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    probs, classes = predict(image_path, model, Max_output)
    print(probs)
    print(classes)
    print("The expexted result is",image_path)