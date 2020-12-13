import os
import cv2
import pandas as pd
import tkMessageBox
import tkSimpleDialog


def callback():
    number = tkSimpleDialog.askinteger("Integer", "Enter your Image_id")
    image_id = number
    # self.root_new.destroy()


def id_default(self):
    # id_count += 1
    image_id = self.id_count
def id_option_more():
    response = tkMessageBox.askyesnocancel("Confirm", "save or not")

    if response == True:

        print("You clicked Yes")
        return "Y"

    elif response == False:

        print("You clicked No")
        return "N"

    elif response is None:
        return "C"

    print("You clicked Cancel")

# def id_option():
#     input_status = tkMessageBox.askquestion("Confirm", "save or not")
#     if input_status == "yes":
#         return "Y"
#     elif:
#         return "N"


bblabel = []
root = "/home/mayank_s/datasets/bdd/bdd100k_images/bdd100k/images/100k/train"
csv_path = '/home/mayank_s/datasets/bdd/training_set/BBD_daytime_train.csv'
data = pd.read_csv(csv_path)
mydata = data.groupby(['filename'], sort=True)

###############################################3333
all_data=data.iloc[:,:].values
x = data.iloc[:, 0].values
y = data.iloc[:, 4:8].values
##################################################
flag=False
loop=0
for dat in all_data:
    loop+=1
    print(loop)
    file_name=dat[0]
    print(file_name)

    if file_name =='0b351bc5-0fd3d464.jpg':
        flag=True
    if flag:
        top=(dat[4],dat[7])
        bottom=(dat[6],dat[5])

        image_path = os.path.join(root, file_name)
        image_scale = cv2.imread(image_path, 1)
        cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        # cv2.circle(image_scale, center=(452, 274), radius=10, color=(0, 255, 0))
        cv2.putText(image_scale, dat[12], (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)

        cv2.imshow('streched_image', image_scale)
        ch = cv2.waitKey(100)
        mychoice = id_option_more()
        print(mychoice)
        if ch & 0XFF == ord('q'):
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()

    #################################################
        (file_name, image_width, image_height, object_name, xmin, ymin, xmax, ymax, bbox_x_normalized,
                      bbox_y_normalized, bbox_width_normalized, bbox_height_normalized, sub_class)=dat
        data_label = [file_name, image_width, image_height, object_name, xmin, ymin, xmax, ymax, bbox_x_normalized,
                      bbox_y_normalized, bbox_width_normalized, bbox_height_normalized, sub_class,mychoice]

        bblabel.append(data_label)
        columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax','bbox_x_normalized', 'bbox_y_normalized', 'bbox_width_normalized', 'bbox_height_normalized','subclass','tl_save_or_not']


        df = pd.DataFrame(bblabel, columns=columns)
        df.to_csv('BBD_user_Select_lights.csv',index=False)
