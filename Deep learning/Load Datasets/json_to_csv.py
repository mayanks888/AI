
import json
import pandas as pd
import os
import cv2

filename=[]
width=[]	
height=[]	
Class=[]	
xmin=[]	
ymin=[]
xmax=[]
ymax=[]
a=[]
file_number=0
#classes=['bus','light','traffic_sign','person','bike','truck','motor','car','train','Rider']
classes=['bus','light','traffic light','person','bike','truck','motor','car','train','Rider',"traffic sign"]
bblabel=[]
loop=0
jason_path="/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/labels/100k/train"
image_path="/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/bdd100k/images/100k/train"

for i in os.listdir(image_path):
    # if loop > 5:
    #     break
    image_name=i
    dat=image_name.split(".")[0]
    filename=os.path.join(jason_path, dat+".json")
    Image_com_path = os.path.join(image_path, image_name)
    try:
        # filename=jason_path+
        # if image_name.split('.')==k.split('.'):
        my_image = cv2.imread(Image_com_path, 1)

        data=open(filename,'r')
        data1 = data.read()
        data.close()
        Json = json.loads(data1)
        # filename.append(Json['name'])
        # for obj in root.iter('object'):
        for ki in Json['frames'][0]['objects']:
            loop+=1
            # print("count run", loop)
            object_name=ki['category']
            if object_name in classes:
                # print(ki.box2d.category)
                xmin=int(ki['box2d']['x1'])
                xmax=int(ki['box2d']['x2'])
                ymin=int(ki['box2d']['y1'])
                ymax=int(ki['box2d']['y2'])
                width=my_image.shape[0]
                height=my_image.shape[1]
                # coordinate = [xmin, ymin, xmax, ymax, class_num]
                data_label = [dat, width, height, object_name, xmin, ymin, xmax, ymax]
                bblabel.append(data_label)
    except IOError:
        print("error at loop:{lp} and image:{dt}".format(lp=loop,dt=dat))
        print('error1')
    except:
        print("error at loop:{lp} and image:{dt}".format(lp=loop, dt=dat))
        print('error2')
        # print('ERROR...object detecion failed for Filename: {fn} , Check file type '.format(fn=filename), '\n')
    else:
        print("successfull for ", dat)


        ''' print(len(Json['frames'][0]['objects']))
            length_Variable=len(Json['frames'][0]['objects'])
            for z in range(length_Variable):
                for j in classes:
                        if j==Json['frames'][0]['objects'][z]['category']:
                            Class.append(Json['frames'][0]['objects'][z]['category'])
                            xmin.append(Json['frames'][0]['objects'][z]['box2d']['x1'])
                            xmax.append(Json['frames'][0]['objects'][z]['box2d']['x2'])
                            ymin.append(Json['frames'][0]['objects'][z]['box2d']['y1'])
                            ymax.append(Json['frames'][0]['objects'][z]['box2d']['y2'])
                            for s in range(len(Class)):
                                b=[filename[file_number]+'.jpg',Class[s],xmin[s],xmax[s],ymin[s],ymax[s]]
                            a.append(b)
                        else:
                            pass
            file_number=file_number+1'''
columns=['filename','Class','xmin','xmax','ymin','ymax']

# pd1=pd.DataFrame(a,columns=columns)
# df=pd.DataFrame(bblabel)
# df.to_csv('out.csv')
# pd1.to_csv('output_bb.csv')


df=pd.DataFrame(bblabel)
df.to_csv('output_Train.csv')
