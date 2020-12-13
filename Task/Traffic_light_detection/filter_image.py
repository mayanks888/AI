import cv2
from scipy.stats import kurtosis, skew

gray = cv2.cvtColor(color_img_crop, cv2.COLOR_BGR2GRAY)
oneDarray = gray.flatten()
kurtValue = kurtosis(oneDarray)
skewValue = skew(oneDarray)
eccentricity = region.eccentricity
