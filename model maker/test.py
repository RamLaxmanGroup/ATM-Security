""" import evaluate_model as ODT    # ODT -> Object Detection

model_path = './models/coin-best.tflite'

INPUT_IMAGE_URL = "./images/yen10.jpg"
DETECTION_THRESHOLD = 0.5 


ODT.main(model_path, INPUT_IMAGE_URL, DETECTION_THRESHOLD) """



""" import cv2
import time
import tensorflow as tf

# reading image
def read_img(frame):
    img=cv2.imread(frame)
    img = tf.io.decode_image(img, channels=3)

start_time = time.time()

cam = cv2.VideoCapture(0)
while True:
    #reading the stream from the selected camera
    ret, frame = cam.read()
    if not ret:
        print("Unable to get video stream.")
        break
    cv2.imwrite('frame.jpg', frame)
    read_img('frame.jpg')
    break

duration = time.time() - start_time
print("Duration", duration)
 """
