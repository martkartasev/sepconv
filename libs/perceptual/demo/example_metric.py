from perceptual.metric import Metric
import cv2

im1 = cv2.imread('images/02-112.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('images/02-149.png', cv2.IMREAD_GRAYSCALE)

m = Metric()

print m.STSIM(im1, im2)
print m.STSIM2(im1, im2)
f =  m.STSIM_M(im1)
print f
print f.shape
