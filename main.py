import cv2
import numpy as np

img = cv2.imread('F1w10bUacAIViQb.jpeg')
h, w, c = img.shape
print(img.shape)

img1 = img[3              : int(h / 2), 0 : w]
img2 = img[int(h / 2)     :      h - 3, 0 : w]

cv2.imwrite("out1.png", img1)
cv2.imwrite("out2.png", img2)

diff = cv2.absdiff(img1, img2)
cv2.imwrite("out.png", diff)

gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
th3 = cv2.adaptiveThreshold(gray_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
      cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("th3.png", th3)

ret, th1 = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("th1.png", th1)

result = cv2.cvtColor(th1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("result.png", result)
