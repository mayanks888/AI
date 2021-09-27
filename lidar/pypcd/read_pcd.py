# import pypcd
from pypcd import pypcd
# import pypcd.pypcd.PointCloud


# also can read from file handles.
v_path = "/media/mayank_sati/New Volume/mayank_linux/roabag/livox/pcd/liv_435699658.pcd"

pc = pypcd.PointCloud.from_path(v_path)
# pc.pc_data has the data as a structured array
# pc.fields, pc.count, etc have the metadata

# center the x field
pc.pc_data['x'] -= pc.pc_data['x'].mean()

# save as binary compressed
pc.save_pcd('bar.pcd', compression='binary_compressed')