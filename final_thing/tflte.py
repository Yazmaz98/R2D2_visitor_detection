import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time
import numpy as np 
import tensorflow as tf
import pyrealsense2 as rs   
from PIL import Image
from arm_detection.yolo3.model import yolo_eval
from arm_detection.yolo3.utils import image_preporcess

import serial
import time
ser = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)

anchors_path='/home/nvidia/Desktop/arm_detection/model_data/yolo_anchors.txt'
classes_path='/home/nvidia/Desktop/arm_detection/4_CLASS_test_classes.txt'
sess = tf.Session()


def getValues(x):
    b_x = bytes(x.encode('utf-8'))
    ser.write(b_x)
    #time.sleep(0.5)
    #arduinoData = ser.readline().decode('ascii')
    return b_x

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
score=0.4
iou=0.3
text_size=1
              # Intel RealSense cross-platform open-source API
hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
np.random.shuffle(colors) 
print("Environment Ready")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(depth_scale)
# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    pipe.wait_for_frames()
frameset = pipe.wait_for_frames()
print("frames detected")
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()
print(depth_frame.get_data())
color = np.asanyarray(color_frame.get_data())
def enum(f):
    return reversed(list(enumerate(f)))
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
num_out=len(output_details)
input_image_shape = tf.placeholder(shape=(2, ),dtype=tf.int32)
outputs=[]
for i in range(num_out):
    outputs.append(tf.placeholder(shape=output_details[i]['shape'],dtype=tf.float32))
boxes_,scores_,classes_=yolo_eval(outputs,anchors, num_classes,input_image_shape,max_boxes=20,score_threshold=score,iou_threshold=iou)

fontScale=1
#video="video_test.mp4"
#try:
#    vid = cv2.VideoCapture(int(video))
#except:
#   vid = cv2.VideoCapture(video)
#width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fps = 30
#codec = cv2.VideoWriter_fourcc(*"XVID")
#out = cv2.VideoWriter("result.mp4", codec, fps, (width, height))
#print(vid)
clipping_distance_in_meters = 10 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
print(input_details)
print(output_details)
while True:
    # set start time to current time
    start_time = time.time()
    # displays the frame rate every 2 second
    display_time = 2
    # Set primarry FPS to 0
    fps = 0
    frameset = pipe.wait_for_frames()
    frameset =align.process(frameset) 
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    depth=np.asanyarray(depth_frame.get_data())
    #ret, frame = vid.read()
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
    frame=  np.asanyarray(color_frame.get_data())
    grey_color = 153
    depth_image_3d = np.dstack((depth,depth,depth)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, frame)
    frame_size=frame.shape[:2]
    image_data =  np.array(image_preporcess(np.copy(frame), (416,416)), dtype=np.float32)
    prev_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    output=[]
    for i in range(num_out):
        output.append(interpreter.get_tensor(output_details[i]['index']))
    
    out_boxes,out_scores,classes=sess.run([boxes_,scores_,classes_],feed_dict={outputs[0]:output[0],outputs[1]:output[1],outputs[2]:output[2],input_image_shape:frame_size})
    thickness = (frame.shape[0] + frame.shape[1]) // 600
    ObjectsList = []
    print(classes)
    for i, c in enum(classes):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
       
        label = '{} {:.2f}'.format(predicted_class, score)
        #label = '{}'.format(predicted_class
        print(label)
        scores = '{:.2f}'.format(score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(frame.shape[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(frame.shape[1], np.floor(right + 0.5).astype('int32'))

        mid_h = (bottom-top)/2+top
        mid_v = (right-left)/2+left
        dist = [np.mean(depth[int(top):int(bottom)+1,int(left):int(right)+1])*depth_scale,depth_frame.get_distance(int(mid_v),int(mid_h)) ]
        #print(depth[int(top):int(bottom)+1,int(left):int(right)+1]*depth_scale)
        #print(depth[int(top):int(bottom)+1,int(left):int(right)+1].shape)
        print(dist[1])
        #getValues(str(round(dist[1],10)))
        getValues(str(int(dist[1]*100)))

        # put object rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), colors[c], thickness)
        # get text size
        (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/text_size, 1)
        # put text rectangle
        cv2.rectangle(frame, (left, top), (left + test_width, top - text_height - baseline), colors[c], thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(frame, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/text_size, (0, 0, 0), 1)

        # add everything to list
        ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])
    curr_time = time.time()
    exec_time = curr_time - prev_time
    #images = np.hstack((frame_color, depth_colormap))
    fps += 1
    TIME = time.time() - start_time
    if TIME > display_time:
        timer="FPS:{:.2f}".format( fps / TIME)
        fps = 0 
        start_time = time.time()
    print(timer)
   # out.write(frame)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result",  frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        vid.release()
        break
cap.release()
sess.close()



