import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

classes = ['hammer', 'mask', 'person', 'smoke', 'knife', 'robbery mask', 'pistol', 'driller', 'keyboard', 'phone', 'screwdriver', 'laptop', 'mouse', 'helmet']
train_data = object_detector.DataLoader.from_pascal_voc('dataset/train', 'dataset/train', classes)
validation_data = object_detector.DataLoader.from_pascal_voc('dataset/test', 'dataset/test', classes)

spec = model_spec.get('efficientdet_lite0')
spec = object_detector.EfficientDetSpec(
  model_name='efficientdet-lite0',
  uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1', 
  hparams={'max_instances_per_image': 500})

model = object_detector.create(train_data, model_spec=spec, epochs = 100, batch_size=16, train_whole_model=True, validation_data=validation_data)
# model.export(export_dir='log_500_8_mm_v2_whole_model')
# model.evaluate(validation_data)




# [[batch_box_loss/write_summary/summary_cond/then/_3398/batch_box_loss/write_summary/ReadVariableOp/_106]]

It resolved! I modified the code to this

# instead of spec = model_spec.get('efficientdet_lite0')

spec = object_detector.EfficientDetSpec(
  model_name='efficientdet-lite0', 
  uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1', 
  hparams={'max_instances_per_image': 8000})

import numpy as np
import os
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)
spec = model_spec.get('efficientdet_lite1')
train_data = object_detector.DataLoader.from_pascal_voc('dataset/train', 'dataset/train', ['log'])
validation_data = object_detector.DataLoader.from_pascal_voc('dataset/test', 'dataset/test', ['log'])
spec = object_detector.EfficientDetSpec(
  model_name='efficientdet-lite1', 
  uri='https://tfhub.dev/tensorflow/efficientdet/lite1/feature-vector/1', 
  hparams={'max_instances_per_image': 500})

model = object_detector.create(train_data, model_spec=spec, epochs = 5, batch_size=16, train_whole_model=True, validation_data=validation_data)
# model.export(export_dir='log_100_32_mm_v2_whole_model')
# model.evaluate(validation_data)