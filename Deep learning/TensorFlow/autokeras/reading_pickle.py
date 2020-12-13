import pickle

# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second_nuscene_mayank/second/save_pkl/nuscenes_infos_train.pkl'
datapath_file = '/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/autokeras/model_file/test_autokeras_model.pkl'
boxes = pickle.load(open(datapath_file, "rb"))
print(1)
