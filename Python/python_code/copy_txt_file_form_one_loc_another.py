import os
import shutil
# input_folder='/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/evaluation/predicition_qatm_thesr_.1_nms.5'
input_folder='/home/mayank_sati/Desktop/tkk/pred'
base_ground_path="/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/evaluation/Ground_truth_val"
new_ground_path="/home/mayank_sati/Desktop/tkk/grd"
output_folder='/home/mayank_sati/Desktop/test_txt'
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
        # return 1
    # time_start = time.time()
    for filename in filenames:
            print("file : {fn}".format(fn=filename), '\n')
            base_ground_file_path = (os.path.join(base_ground_path, filename))
            new_ground_file_path=(os.path.join(new_ground_path, filename))
            shutil.copyfile(base_ground_file_path, new_ground_file_path)
'''            
            ###############################3333
            labelname = (os.path.join(root, filename))
            output_labelfilename = (os.path.join(output_folder, filename))

            # labelname = imagename + '.txt'
            # labelfilename = os.path.join(outDir, labelname)
            bbox_cnt = 0
            if os.path.exists(labelname):
                with open(output_labelfilename, 'w') as k:
                    with open(labelname) as f:
                        for (i, line) in enumerate(f):
                            yolo_data = line.strip().split()

                            # yolo_data[1]=str(int(float(yolo_data[1]))/2)
                            # yolo_data[2]=str(int(float(yolo_data[2]))/2)
                            # yolo_data[3]=str(int(float(yolo_data[3]))/2)
                            # yolo_data[4]=str(int(float(yolo_data[4]))/2)
                            #
                            # xmin = int(float(yolo_data[1]) / 2)
                            # ymin = int(float(yolo_data[2]) / 2)
                            # xmax = int(float(yolo_data[3]) / 2)
                            # ymax = int(float(yolo_data[4]) / 2)

                            yolo_data[1] = int(float(yolo_data[1]) / 2)
                            yolo_data[2] = int(float(yolo_data[2]) / 2)
                            yolo_data[3] = int(float(yolo_data[3]) / 2)
                            yolo_data[4] = int(float(yolo_data[4]) / 2)
                            # tmp = deconvert(yolo_data[1:])
                            # tmp = [int(float(loop)) for loop in yolo_data[1:-1]]
                            # tmp = [int(float(loop)) for loop in yolo_data[1:-1]]
            ####################################
                            # bb = ((xmin), (ymin), (xmax), (ymax))
                            k.write(" ".join([str(a) for a in yolo_data])+'\n')
                            # f.write(str('traffic_light') + " " + " ".join([str(a) for a in bb])+'\n')

                            # f.write(" ".join([str(a) for a in yolo_data]) + '\n')
                            # f.write(" ".join([str(a) for a in yolo_data])+ '\n')
            # output_path = (os.path.join(output_folder, filename))
            # cv2.imwrite(output_path, img)
'''