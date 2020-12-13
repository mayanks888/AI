import autokeras as ak
from autokeras.image.image_supervised import load_image_dataset

clf = ak.ImageClassifier(verbose=True, augment=False)
train_image_path = "/home/mayank_sati/Desktop/sorting_light/all_train_images"
test_image_path = "/home/mayank_sati/Desktop/sorting_light/all_train_images"
train_csv = "/home/mayank_sati/Desktop/sorting_light/train_color_autokeras.csv"
test_csv = "/home/mayank_sati/Desktop/sorting_light/test_color_autokeras.csv"
x_train, y_train = load_image_dataset(csv_file_path=train_csv,
                                      images_path=train_image_path)
print(x_train.shape)
print(y_train.shape)
x_test, y_test = load_image_dataset(csv_file_path=test_csv, images_path=train_image_path)
print(x_test.shape)
print(y_test.shape)

# clf = ImageClassifier(verbose=True)
clf.fit(x_train, y_train, time_limit=1 * 60 * 60)
clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
y = clf.evaluate(x_test, y_test)
print(y)

###########################################################3
# for exporting model
# clf.export_autokeras_model(model_file_name)
# from autokeras.utils import pickle_from_file
# model = pickle_from_file(model_file_name)
# results = model.evaluate(x_test, y_test)
# print(results)

#########################################################################
# # for visualising
# pip:  pip install graphviz
#
# conda : conda install -c conda-forge python-graphviz
# If the above installations are complete, proceed with the following steps :
#
# Step 1 : Specify a path before starting your model training
#
#
# clf = ImageClassifier(path="~/automodels/",verbose=True, augment=False) # Give a custom path of your choice
# clf.fit(x_train, y_train, time_limit=30 * 60)
# clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
# Step 2 : After the model training is complete, run examples/visualize.py, whilst passing the same path as parameter
#
#
# if __name__ == '__main__':
#     visualize('~/automodels/')
