import os

import cv2
import numpy as np
import tensorflow as tf
from autokeras.utils import pickle_from_file

if __name__ == '__main__':
    model_file_name = '/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/autokeras/model_file/test_autokeras_model_2.pkl'
    model = pickle_from_file(model_file_name)
    # results = model.evaluate(x_test, y_test)
    # model = my_model('/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/saved_models/model_f_lr-4_ep50_ba32.h')
    # model = load_model('/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/saved_models/model_b_lr-4_ep50_ba32.h')
    # model = load_model('/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/saved_models/model_b_lr-4_ep150_ba32.h')
    # model = build_model()
    # lr=.001
    # opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.load_weights('first_try2.h5')
    # model.load_weights('first_try2.h5')

    ##################################################33333
    input_folder = "/home/mayank_sati/Desktop/sorting_light/all_train_images"
    traffic_light = ["red", "green", "black"]
    for root, _, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
        # time_start = time.time()
        for filename in filenames:
            filename = input_folder + "/" + filename
            test_image = tf.keras.preprocessing.image.load_img(path=filename, target_size=(55, 35))
            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            # test_image /=255
            result = model.predict(test_image)
            print(result)
            # light_color=traffic_light[result.argmax()]
            light_color = traffic_light[int(result[0])]
            # dat = (training_data.class_indices)
            # res = ({v: k for k, v in dat.items()}[result.argmax()])
            # print(res)

            image_scale = cv2.imread(filename, 1)
            cv2.putText(image_scale, light_color, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 255, 0),
                        lineType=cv2.LINE_AA)
            cv2.imshow('streched_image', image_scale)
            # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
            # filepath=output_folder_path+my_image+".png"
            # cv2.imwrite(filepath,my_image)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(image_scale, z[val], (100, 200), font, 4, (0, 255, 0), 2, cv2.LINE_AA)
            # if val > 185:
            #     print(1)
            # cv2.waitKey(1000)
            ch = cv2.waitKey(2000)  # refresh after 1 milisecong
            if ch & 0XFF == ord('q'):
                cv2.destroyAllWindows()
            cv2.destroyAllWindows()
    ##########################################################
