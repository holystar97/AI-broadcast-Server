import cv2
import numpy as np
import glob

PATH = "/Users/mkrice/Desktop/REGALA/mqtt-camera-streamer-master/captures"
img_array = []
for filename in glob.glob(f'{PATH}/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


out = cv2.VideoWriter('/Users/mkrice/Desktop/REGALA/iPortfolio/static/video/project2.mp4',
                      cv2.VideoWriter_fourcc(*'H264'), 3, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
