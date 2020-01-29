# R2D2_visitor_detection
This is for a Mechatronics project at Emines School of Industrial Management - UM6P in Morocco. The robot must detect when a 
visitor enters the room, then move towards them and ask if they need assistance.

## Training and detections on your laptop

### Libraries versions for compatibility purposes

  - tensorflow==1.14.0 
  - Keras==2.2.4
  - python==3.6
  - yolo3

### I/ Preparing the Dataset

1. Download the images you're going to train your model on from a cloud datasets platform or from google images using "Fatkun" 
extension. Put them in OID/Dataset/train/name_of_your_class.
2. Download LabelImg from https://github.com/tzutalin/labelImg and label your images manually 
(or use ***oid_to_pascal_voc_xml.py*** if you've downloaded your dataset from Google cloud)
3. You must now have a corresponding .xml file for each one of the images you've labeled, containing the coordinates of the 4 
points constituting the boxes containing your object.
4. Run ***voc_to_YOLOv3.py***; 2 .txt files should be generated: 4_CLASS_test.txt and 4_CLASS_test_classes.txt
-------------------------------------------------------------------------------------------------------------------------------
                                    CONGRATULATIONS! You may now start your training.
-------------------------------------------------------------------------------------------------------------------------------
### II/ Training

In order to start training

5. First go to https://pjreddie.com/darknet/yolo/ and download yolov3.weights and yolov3.cfg <br/>
(you might find a number in front of yolov3 for these two files, this number determines the number of classes pjreddie has 
trained his models on).
6. Run the following line of code:<br/>
        "python3 convert.py model_data/yolov3.weights model_data/yolov3.cfg model_data/yolo_weights.h5"<br/>
this should transform the yolov3 weights into a .h5 file that we'll use during our training. 
7. Open ***train.py***, make the modifications that you need to make (for beginners this could be the batch_size and number of 
epochs, remember: this is a trial and error procedure, these parameters will depend on the size of your dataset)
8. Your training will be logged in the /logs folder. Use the obtained .h5 file in ***screen_detect.py*** if you want to try out 
detections on your monitor (google search engine for example), or ***webcam_detect.py*** if you want to use your laptop's camera.<br/>
ATTENTION: if you do not have a gpu comment out the "OS.DEVICE....etc".

Training might think a couple of hours (depending on the size of your dataset, batch size, and epochs).

## Detections on the NVIDIA Jetson TX2

On the NVIDIA Jetson, we've decided to use TFlite instead of yolo or tiny-yolo.

### Libraries versions for compatibility purposes

  - tensorflow==2.0 
  - Keras==2.3.1
  - python==3.6
 
 ### Steps to make your previous .h5 model work on the NVIDIA Jetson tx2
 
 1. Run ***tlite-test.py*** to convert the .h5 weights file into a tensorflow-lite supported model.<br/>
 2. The output is ***converted_model.tflite***
 3. Run ***tflte.py***
 
 ***ATTENTION***
 
 After every detection, the ***tflte.py*** file then sends a command to an Arduino board via the Serial line to move towards 
 the visitor.
