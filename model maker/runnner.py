# uses the function from object_detector.py file
# run this file to start object dectection on frames
# from webcam

import obj_detector as ODT
import config
import cv2
import numpy as np
from PIL import Image
import time

def get_params():
    params = dict()
    
    #params["model_path"] = './models/atm_74_8_lite0_mm/atm_74_8_lite0_mm.tflite'
    #params["model_path"] = './models/atm_150_16_lite0_mm/atm_150_16_lite0_mm.tflite'
    #params["model_path"] = './models/atm_173_4_lite1_mm/atm_173_4_lite1_mm.tflite'
    params["model_path"] = './models/atm_v2_105_4_lite1_mm/atm_v2_105_4_lite1_mm.tflite'

    params["img_path"] = "../test images/test.jpg"
    params["frame_path"] = "./images/frame.jpg"
    params["detect_threshold"] = 0.5
    
    return params


if __name__=="__main__":
    params = get_params()
    cam = cv2.VideoCapture(0)
    while True:
        start_time = time.time()
        #reading the stream from the selected camera
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            print("Unable to get video stream.")
            break
        #conveting to PIL Image
        #img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cv2.imwrite(params["img_path"], frame)
        
        ODT.main(params["model_path"], params["img_path"], params["detect_threshold"])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #duration = time.time() - start_time
        #print("Duration", duration)
    cam.release()
    cv2.destroyAllWindows()
