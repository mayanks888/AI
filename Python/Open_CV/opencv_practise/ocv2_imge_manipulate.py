import cv2

img=cv2.imread("car.jpg",1)
print(img.shape)
print(img.size)#total pixel
b, g, r=cv2.split(img)
mer=cv2.merge((b,g,r))

########################3333
crop_image=img[20:70,10:50]#[1,3,0,2][ymin:ymax, xmin:xmax]
# image_scale[int(lis[1]):int(lis[3]), int(lis[0]):int(lis[2])]

##################################3
#kind of watermark
# add two images
img2=cv2.imread("img.png")
img3=cv2.add(img,img2)
img3=cv2.addWeighted(img,.5,img2,.5,0)
#################################
cv2.imshow("image",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()