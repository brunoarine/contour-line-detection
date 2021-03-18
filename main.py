# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
from aux import skeleton
from gabor import gabor_filter

image = cv2.imread('resources/talhao.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
binary = cv2.adaptiveThreshold(image_rgb[:,:,0],255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,119,8)

cv2.imwrite("binary.jpg", binary)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))

gabor = gabor_filter(binary)
erosion = cv2.erode(gabor, kernel)
dilation = cv2.dilate(erosion, kernel, iterations=2)
cv2.imwrite("morph.jpg", dilation)

skel = skeleton(dilation)
cv2.imwrite("skeleton.jpg", skel)


threshold = 50
maxLineGap = 5
minLineLength = 20
lines = cv2.HoughLinesP(skel, 1, np.pi/180, threshold=threshold,
                  maxLineGap=maxLineGap,
                  minLineLength=minLineLength);
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),5)
cv2.imwrite("lines.jpg", image)




#plt.figure(figsize=(10, 7))

#plt.subplot(121),plt.imshow(image_rgb)
#plt.subplot(122),plt.imshow(binary)
#plt.show()
