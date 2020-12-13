import glob, os

label_dir = '/home/mayank_s/datasets/bdd/training_set/one_class/train_bdd_front_light'
image_dir = '/home/mayank_s/datasets/bdd/bdd100k_images/bdd100k/images/100k/train'

for label in os.listdir(label_dir):
    if label.endswith('.txt'):
        label_name = os.path.splitext(label)[0]

        # Corresponding label file name
        image_name = label_name + '.jpg'
        if os.path.isfile(image_dir + '/' + image_name) == False:
            print(" -- DELETE LABEL [Image file not found -- ]")
            label_path = label_dir + '/' + label_name + '.txt'
            os.remove(label_path)
