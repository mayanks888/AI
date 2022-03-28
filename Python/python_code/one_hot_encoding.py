import numpy as np
nb_classes = 11
# targets = np.array([[2, 3, 4, 0]]).reshape(-1)
targets = np.array([2]).reshape(-1)
one_hot_targets = np.eye(nb_classes)[targets]
repeats_array = np.tile(one_hot_targets, (2, 1))
1

# modified by mayank for adding label one hot encodng
class_categories = ['backgound', 'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier']
# label_data = class_categories.index("truck")
label_data =  np.array(class_categories.index("truck")).reshape(-1)
nb_classes = 11
one_hot_targets1 = np.eye(nb_classes)[label_data]
# one_hot_targets[:, 0] = 0
repeats_array1 = np.tile(one_hot_targets1, (2, 1))
1