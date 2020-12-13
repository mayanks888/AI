import cv2

from PIL import Image, ImageDraw

image = cv2.imread("/home/mayank_sati/Desktop/farm_temp_sample/1.jpg", 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

im = Image.fromarray(image)
# im = Image.new('RGB', (500, 300), (128, 128, 128))
draw = ImageDraw.Draw(im)
draw.rectangle((200, 100, 300, 200), fill=(0, 0, 0), outline=(255, 255, 255))
# draw.rectangle((200, 100, 300, 200), fill="#ffff33", outline=(255, 255, 255))
# image.rectangle(shape, fill="# ffff33", outline="green")
im.show()
