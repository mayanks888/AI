import csv
import os

# train_dir = '/home/mayank_sati/Downloads/load_raw_image/train' # Path to the train directory
# train_dir = '/home/mayank_sati/Desktop/sorting_light/complte_data' # Path to the train directory
train_dir = '/home/mayank_sati/Desktop/sorting_light/complete_image_with_diff_name'  # Path to the train directory
class_dirs = [i for i in os.listdir(path=train_dir) if os.path.isdir(os.path.join(train_dir, i))]
1
csv_path = '/home/mayank_sati/Desktop/train_color_autokeras.csv'
# with open('train/label.csv', 'w') as train_csv:
with open(csv_path, 'w') as train_csv:
    fieldnames = ['File Name', 'Label']
    writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
    writer.writeheader()
    label = 0
    for current_class in class_dirs:
        for image in os.listdir(os.path.join(train_dir, current_class)):
            writer.writerow({'File Name': str(image), 'Label': label})
        label += 1
    train_csv.close()
