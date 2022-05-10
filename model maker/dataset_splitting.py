# Run this file from Anaconda Promt
import os, random, shutil
try:
    os.mkdir('dataset/train')
    os.mkdir('dataset/test')
except:
    print("train test directory already exits.")


image_paths = os.listdir('dataset/images')
random.shuffle(image_paths)


for i, image_path in enumerate(image_paths):
  if i < int(len(image_paths) * 0.8):
    try:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/train')
        if image_path.startswith('armas'):
            image_path = image_path[:-4] + '.xml'
            shutil.copy(f'dataset/annotations/{image_path}', 'dataset/train')
        else:
            shutil.copy(f'dataset/annotations/{image_path.replace("jpg", "xml")}', 'dataset/train')
    except:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/train')
        
        if image_path.startswith('armas'):
            image_path = image_path[:-4] + '.xml'
            shutil.copy(f'dataset/annotations/{image_path}', 'dataset/train')
        else:
            shutil.copy(f'dataset/annotations/{image_path.replace("JPG", "xml")}', 'dataset/train')

  else:
    try:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/test')
        if image_path.startswith('armas'):
            image_path = image_path[:-4] + '.xml'
            shutil.copy(f'dataset/annotations/{image_path}', 'dataset/test')
        else:
            shutil.copy(f'dataset/annotations/{image_path.replace("jpg", "xml")}', 'dataset/test')
    except:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/test')
        if image_path.startswith('armas'):
            image_path = image_path[:-4] + '.xml'
            shutil.copy(f'dataset/annotations/{image_path}', 'dataset/test')
        else:
            shutil.copy(f'dataset/annotations/{image_path.replace("JPG", "xml")}', 'dataset/test')