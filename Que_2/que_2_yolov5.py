# -*- coding: utf-8 -*-
"""Que_2_YoloV5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kuR62yeMBONda2u46WnXcCf_Bu51N988
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  # clone
# %cd yolov5
# %pip install -qr requirements.txt  # install

import torch
import utils
display = utils.notebook_init()  # checks
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Train YOLOv5s on dataset
!python train.py --img 416 --batch 16 --epochs 70 --data /content/drive/MyDrive/Neetiraj_Assignment/Dataset/data.yaml --weights yolov5s.pt --cache

!python detect.py --weights runs/train/exp3/weights/best.pt --img 416 --conf 0.25 --source /content/drive/MyDrive/Neetiraj_Assignment/Dataset/test/images

import glob
from IPython.display import Image, display
for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'):
    display(Image(filename=imageName))
    print("\n")

"""**Visualize**

for exprement 3 where batchsize 16 and epoch 70
"""

Image('/content/yolov5/runs/train/exp3/F1_curve.png')

Image('/content/yolov5/runs/train/exp3/confusion_matrix.png')

Image('/content/yolov5/runs/train/exp3/train_batch0.jpg')

Image('/content/yolov5/runs/train/exp3/train_batch2.jpg')

