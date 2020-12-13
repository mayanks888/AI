import pandas as pd
import yaml

filename = []
width = 0
height = 0
Class = []
xmin = []
ymin = []
xmax = []
ymax = []
light_color = []
bblabel = []
loop = 1
image_path = "/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-25-36_copy/GWM_dataset"
object_name = "traffic_light"
yaml_path = "/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-25-36_copy/mayank_first.yaml"
with open(yaml_path, 'r') as ymlfile:
    # with open("/home/mayanksati/PycharmProjects/models/tensorflow/re3-tensorflow-master/demo/mayank_first.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

for section in cfg:
    print(section)
    loop += 1
    xmax = section['boxes'][0]['boxes']['xmax']
    xmin = section['boxes'][0]['boxes']['xmin']
    ymax = section['boxes'][0]['boxes']['ymax']
    ymin = section['boxes'][0]['boxes']['ymin']
    height = section['boxes'][0]['boxes']['height']
    width = section['boxes'][0]['boxes']['width']
    # ___________________________________________
    # playing with file
    file_path = section['path']
    file_name = file_path.split("/")[-1]
    new_path = image_path + "/" + file_name
    file_path = new_path
    # my_image = cv2.imread(new_path, 1)
    # # ++++++++++++++++++++++++++++++++++++++++++++++=
    # new_filename= "baidu_version_2_"+str(loop)+ ".jpg"
    # output_path =new_image_path + new_filename
    # print(output_path)
    # # try:
    # cv2.imwrite(output_path, my_image)
    # # ++++++++++++++++++++++++++++++++++++++++++++++++
    # width = my_image.shape[1]
    #
    # height = my_image.shape[0]

    data_label = [file_path, width, height, object_name, xmin, ymin, xmax, ymax]
    if not ((xmin == xmax) and (ymin == ymax)):
        bblabel.append(data_label)
        # print()
    # if loop>=1018:
    #     break

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

df = pd.DataFrame(bblabel, columns=columns)
print("into csv file")
df.to_csv('square3_3.csv', index=False)
