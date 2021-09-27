import numpy as np
# npy_files='/home/mayank_sati/Documents/lidar_data/mayank_wieng_trt/rpn_2/rpn_21_05_2021_17_17_03_820310.npy'
npy_files='/home/mayank_sati/Documents/lidar_data/maynk_base_model/pfn_linear_fixed_inp_conv/pfn_24_05_2021_20_03_31_031730.npy'
# npy_files='/home/mayank_sati/Documents/lidar_data/maynk_base_model/pfn/pfn_03_05_2021_19_49_28_919777.npy'
feat_val = np.load(npy_files)
feat_val = feat_val.astype('uint8')
feat_val=np.expand_dims(feat_val, axis=2)
feat_val=np.transpose(feat_val,(0,3,1,2))
print(feat_val.shape)
(5164,9,60,1)
# return feat_val
