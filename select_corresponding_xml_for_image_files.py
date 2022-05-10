from __future__ import annotations
import shutil
import os

# Directory to the image files
# first replacing \v with /v as \v together provides another character
# then replcing backslash with forward slash
img_dir = 'D:\Anaconda\ATM Security\image data\v2\images'.replace("\v", "/v").replace("\\","/")
annotations_dir = "D:\Anaconda\ATM Security\image data\annotations".replace("\a", "/a").replace("\\","/")
dst_annotation_dir = 'D:/Anaconda/ATM Security/image data/v2/annotations'

if not os.path.exists(dst_annotation_dir):
    try:
        os.mkdir(dst_annotation_dir)
    except:
        print(f"Directory {dst_annotation_dir} cannot be created.")
        exit()

# getting all the filenames of the given directory in list
filenames = os.listdir(img_dir)
count = 0
# get annotation files corresponding to the image files.
for filename in filenames:
    filename = filename.split('.')[0] # removing ".jpg" file extention from filename
    filename = filename + '.xml' # adding ".xml" file extenstion to filename
    filename=os.path.join(annotations_dir, filename)
    shutil.copy(filename, dst_annotation_dir)
    count += 1

print(f"Copied {count} files.")