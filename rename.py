from itertools import count
import os
import cv2
import random
dataset_path = "./image data/v2/raw"
new_dataset_path = "./image data/v2/images"

if not os.path.exists(new_dataset_path):
        print(f"Directory unavailable, Creating directory {new_dataset_path}")
        os.mkdir(new_dataset_path)

files = os.listdir(dataset_path)
#print(len(files))
random.shuffle(files)

for index, file in enumerate(files):
    # update the index value if new files are to added in ./image data/remaned/ directory.
    img_path = os.path.join(dataset_path, file)
    img = cv2.imread(img_path)
    project_name='atm_v2_'
    file_name=project_name + str(index) + '.jpg'
    
    if file_name.endswith('.jpg'):
        new_img_path = os.path.join(new_dataset_path, file_name)
        cv2.imwrite(new_img_path, img)
        print(new_img_path + " Is saved in new directory.")