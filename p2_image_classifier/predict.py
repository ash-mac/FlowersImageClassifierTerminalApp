import argparse
import warnings
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os
import logging 

warnings.filterwarnings("ignore")

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(
    description='Arguments in order are: ImageAddressInCurrentDirectory ModelAddress TopKPrediction ClassNames',
)
parser.add_argument('image_path')
parser.add_argument('model_path')
parser.add_argument('--top_k',action = 'store',dest = 'top_k',type = int,default = 1,help = 'gives top k possibilities of flower')
parser.add_argument('--category_names',action = 'store',dest = 'json_file',help = 'lists all flowers in the dataset')
args = parser.parse_args()

im_path = args.image_path
model_path = args.model_path
top_k = args.top_k
json_file = args.json_file

import json
with open('label_map.json', 'r') as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(("./"+model_path), custom_objects={'KerasLayer':hub.KerasLayer})

def process_image(image):
    only_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    only_tensor = tf.image.resize(only_tensor, (224, 224))
    only_tensor/=255
    image = only_tensor.numpy()
    return image

def predict(image_path,model,k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image,axis = 0)
    predictions = model.predict(image,batch_size = None)
    top_k_probs, top_k_indices = tf.nn.top_k(predictions, k=k)
    top_k_probs = list(top_k_probs.numpy()[0])
    top_k_classes =[]
    for index in top_k_indices[0]:
      top_k_classes.append(class_names[str(index.numpy()+1)])
    return top_k_probs,top_k_classes
import pprint

probabilities,flowers = (predict(("./"+im_path),model,top_k))

n = len(probabilities)

for i in (range(n)):        
     print('{} => {} with probability = {}'.format(i+1,flowers[i],probabilities[i]))

if(json_file != None):
    if(os.path.isfile(json_file)):
        with open(json_file, 'r') as f:
            class_mapping = json.load(f)
        pprint.pprint(class_mapping)
    else:
        print('Invalid file address!')
    