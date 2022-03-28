import pickle

# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second_nuscene_mayank/second/save_pkl/nuscenes_infos_train.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/autokeras/model_file/test_autokeras_model.pkl'
# datapath_file = '/home/mayank_sati/pycharm_projects/pytorch/second_nuscene_mayank/second/pytorch/apollo_pp.pkl'
# datapath_file1 = '/home/mayank_sati/Downloads/v1.0-mini/infos_train_10sweeps_withvelo_filter_True.pkl'
datapath_file1 = '/home/mayank_sati/Downloads/v1.0-mini/infos_train_11sweeps_withvelo_filter_True.pkl'
datapath_file2 = '/media/mayank_sati/wd_2tb/myjob/datasets/detection/nuscenes/v1.0-mini/infos_train.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second .pytorch_traveller59_date_9_05_experiment_mode/second/pytorch/models/mayank_rpn.onnx'
boxes1 = pickle.load(open(datapath_file1, "rb"))
boxes2 = pickle.load(open(datapath_file2, "rb"))
print(1)
# import mayavi.mlab as mlab
#
# fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
#         pcd_data = gt_points
#         print(pcd_data.shape)
#         # pcd_data = points
#         draw_lidar(pcd_data, fig=fig)
#         mlab.show()
