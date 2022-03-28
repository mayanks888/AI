import json
import pickle
file_path='/home/mayank_sati/Downloads/results_nusc.json'
# data =json.load(file_path)

data = open(file_path, 'r')
data1 = data.read()
data.close()
Json = json.loads(data1)
1
pickle_path='/home/mayank_sati/Downloads/result.pkl'


pickle_in = open(pickle_path, "rb")
result_pickle_data = pickle.load(pickle_in)


pickle_path_info='/home/mayank_sati/Downloads/v1.0-mini/infos_val.pkl'


pickle_in1 = open(pickle_path_info, "rb")
info_val_data = pickle.load(pickle_in1)
1