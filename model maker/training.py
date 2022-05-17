import numpy as np
import os
import time

#from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# function to make directory of the given path
def make_dir(path):
    # check if path exists in local directory
    if not os.path.exists(path):
        print(f'Creating directory {path}')
        os.mkdir(path)      # creating directory
    else:
        print(f'Directory {path} Already Exists. Cannot Overwrite.')
        exit()      # stopping the program to run to control the overwriting of the checkpoints, saved model and tflite.

# lite model
lite = 'lite0'
suffix = 'temp2'

# dictionary with paths
dir_dict = {}
root = f'real_atm_v1_{lite}_best_16_{suffix}'
dir_dict['export_dir'] = f'./models/{root}'
dir_dict['model_dir'] = f'./models/{root}/checkpoints/'
dir_dict['saved_model_dir'] = f'./models/{root}/saved_model/'

# iterating over the keys of dictionary
for key in dir_dict:
    # calling function to make directory if directory doesn't exists.
    make_dir(dir_dict[key])
    

# select model architecture
#spec = model_spec.get('efficientdet_lite0')

spec = object_detector.EfficientDetSpec(
    model_name=f'efficientdet-{lite}', 
    uri=f'https://tfhub.dev/tensorflow/efficientdet/{lite}/feature-vector/1', 
    tflite_max_detections = 25, # number of objects to be detected in image
    # adjusitng the hyperparameters
    hparams={'max_instances_per_image': 30, 
        'optimizer': 'sgd',  # adam, sgd, 
        'lr_decay_method': 'cosine', # cosine, stepwise, polynomial
        'model_dir' : dir_dict['model_dir'],
        'saved_model_dir' : dir_dict['saved_model_dir']
        })    # sgd and cosine combination works well

# Load the dataset
labels = ['hammer', 'mask', 'person', 'smoke', 'knife', 'pistol', 'driller', 'keyboard', 'phone', 'screwdriver', 'laptop', 'mouse', 'helmet']
train_data = object_detector.DataLoader.from_pascal_voc(images_dir='dataset/train', annotations_dir='dataset/train', label_map=labels)
validation_data = object_detector.DataLoader.from_pascal_voc(images_dir='dataset/test', annotations_dir='dataset/test', label_map=labels)

# Training of tensorflow lite model 
model = object_detector.create(train_data, model_spec=spec, epochs = 1000, batch_size=16, train_whole_model=True, validation_data=validation_data, do_train=True)

time.sleep(10)
# Model evaluation on test data
model.evaluate(validation_data)

# Export tensorflow model
# './directory/data-type_project_version_architecture_epochs_batch-size'
model.export(export_dir=dir_dict['export_dir'])

