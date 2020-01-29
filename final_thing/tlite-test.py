import os
import numpy as np 
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from arm_detection.yolo3.model import yolo_body
anchors_path='/home/nvidia/Desktop/arm_detection/model_data/yolo_anchors.txt'
classes_path='/home/nvidia/Desktop/arm_detection/4_CLASS_test_classes.txt'
def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
anchors=get_anchors(anchors_path)
def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
class_names=get_class(classes_path)
num_anchors = len(anchors)
num_classes = len(class_names)
score=0.3
iou=0.4
model=yolo_body(Input(shape=(416,416,3)), num_anchors//3, num_classes)
model .load_weights('/home/nvidia/Desktop/arm_detection/logs/arm_trained/trained_weights_final.h5')
print(model.outputs)
def enum(f):
    return reversed(list(enumerate(f)))
with K.get_session() as sess: 
   converter = tf.lite.TFLiteConverter.from_session(sess,model.inputs, model.outputs)
   tflite_model = converter.convert()
   open("converted_model.tflite", "wb").write(tflite_model)  
   
