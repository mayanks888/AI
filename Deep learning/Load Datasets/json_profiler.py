import json

import pandas as pd

name = []
dur = []
time = []

a = []
file_number = 0
# classes=['bus','light','traffic_sign','person','bike','truck','motor','car','train','Rider']
classes = ['bus', 'light', 'traffic light', 'person', 'bike', 'truck', 'motor', 'car', 'train', 'Rider', "traffic sign"]
bblabel = []
loop = 0
jason_path = "/home/mayanksati/Documents/datasets/BDD/bdd100k/images/100k/train"
image_path = "/home/mayanksati/Documents/datasets/BDD/bdd100k/images/100k/train"

# for i in os.listdir(image_path):
#     # if loop > 5:
#     #     break
#     image_name=i
#     dat=image_name.split(".")[0]
#     filename=os.path.join(jason_path, dat+".json")
#     Image_com_path = os.path.join(image_path, image_name)
# try:
# filename=jason_path+
# if image_name.split('.')==k.split('.'):
# my_image = cv2.imread(Image_com_path, 1)
filename = '/home/mayanksati/Desktop/mayank_profiler/without crop/json/crop_img_4.jpg.json'
data = open(filename, 'r')
data1 = data.read()
data.close()
Json = json.loads(data1)
# filename.append(Json['name'])
# for obj in root.iter('object'):
# for ki in Json['frames'][0]['objects']:
cool = Json['traceEvents']
for ki in cool:
    loop += 1
    # print("count run", loop)
    # object_name=ki['category']
    print(ki['name'])
    if ki['name'] == "Conv2D":
        # print(ki.box2d.category)
        dur = int(ki['dur'])
        name = ki['args']['name']
        time = int(ki['ts'])
        # xmax=int(ki['box2d']['x2'])
        # coordinate = [xmin, ymin, xmax, ymax, class_num]
        data_label = [name, dur, time]
        bblabel.append(data_label)
# except IOError:
#     print("error at loop:{lp} and image:{dt}".format(lp=loop,dt=1))
#     print('error1')
# except:
#     print("error at loop:{lp} and image:{dt}".format(lp=loop, dt=1))
#     print('error2')
#     # print('ERROR...object detecion failed for Filename: {fn} , Check file type '.format(fn=filename), '\n')
# else:
#     print("successfull for ", 1)


columns = ['operation name', 'duration', 'time']

# pd1=pd.DataFrame(a,columns=columns)
# df=pd.DataFrame(bblabel)
# df.to_csv('out.csv')
# pd1.to_csv('output_bb.csv')


df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('profile_fasterrcnn.csv')
