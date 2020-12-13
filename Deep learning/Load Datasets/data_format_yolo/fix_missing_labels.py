import glob, os

'''
Sometimes your image data set might not match with your label data set.
This code does the folowing
(1) Go through your image data set
(2) Search if the corresponding label file exist in the label data set. 
(3) If not, remove current image
'''

label_dir = '/home/mayank_s/datasets/bdd/training_set/one_class/train_bdd_front_light'
image_dir = '/home/mayank_s/datasets/bdd/training_set/one_class/bdd100k_images/bdd100k/images/100k/train'

for image in os.listdir(image_dir):
    if image.endswith('jpg'):
        image_name = os.path.splitext(image)[0]

        # Corresponding label file name
        label_name = image_name + '.txt'
        if os.path.isfile(label_dir + '/' + label_name) == False:
            print(" -- DELETE IMAGE [Label file not found -- ]")
            image_path = image_dir + '/' + image_name + '.jpg'
            os.remove(image_path)
