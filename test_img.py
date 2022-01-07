import cv2
from img_processing_FX.output_img_refactor import img_procesings

img = cv2.imread("./img_processing_FX/ana_green_sable.png")
img_procesings(img)
