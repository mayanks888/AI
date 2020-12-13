import numpy as np
import os
import sys

file_path_gt = "./val_output_gt_2019-09-13-14-15-09.txt"

count_gt = 0
count_car_dist_gt = np.zeros(4)
count_van_dist_gt = np.zeros(4)
count_truck_dist_gt = np.zeros(4)
count_pedestrian_dist_gt = np.zeros(4)
count_tram_dist_gt = np.zeros(4)
count_cyclist_dist_gt = np.zeros(4)
count_misc_dist_gt = np.zeros(4)
count_person_sitting_dist_gt = np.zeros(4)

count = 0
count_car_dist = np.zeros(4)
count_van_dist = np.zeros(4)
count_truck_dist = np.zeros(4)
count_pedestrian_dist = np.zeros(4)
count_tram_dist = np.zeros(4)
count_cyclist_dist = np.zeros(4)
count_misc_dist = np.zeros(4)
count_person_sitting_dist = np.zeros(4)

count_car_size = 0
count_van_size = 0
count_truck_size = 0
count_pedestrian_size = 0
count_tram_size = 0
count_cyclist_size = 0
count_misc_size = 0
count_person_sitting_size = 0

short_dist = 10.0
mid_dist_1 = 20.0
mid_dist_2 = 30.0
print("****************************************************")

if os.path.exists(file_path_gt):
    with open(file_path_gt) as file:
        for line_num, line_ in enumerate(file):
            line_ = line_[:-1].split(' ')
            line_tmp = line_[1:]

            if line_tmp[1] == "car":
                if float(line_tmp[2]) <= short_dist:
                    count_car_dist_gt[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_car_dist_gt[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_car_dist_gt[2] += 1
                else:
                    count_car_dist_gt[3] += 1
            elif line_tmp[1] == "van":
                if float(line_tmp[2]) <= short_dist:
                    count_van_dist_gt[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_van_dist_gt[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_van_dist_gt[2] += 1
                else:
                    count_van_dist_gt[3] += 1
            elif line_tmp[1] == "truck":
                if float(line_tmp[2]) <= short_dist:
                    count_truck_dist_gt[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_truck_dist_gt[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_truck_dist_gt[2] += 1
                else:
                    count_truck_dist_gt[3] += 1
            elif line_tmp[1] == "pedestrian":
                if float(line_tmp[2]) <= short_dist:
                    count_pedestrian_dist_gt[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_pedestrian_dist_gt[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_pedestrian_dist_gt[2] += 1
                else:
                    count_pedestrian_dist_gt[3] += 1
            elif line_tmp[1] == "tram":
                if float(line_tmp[2]) <= short_dist:
                    count_tram_dist_gt[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_tram_dist_gt[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_tram_dist_gt[2] += 1
                else:
                    count_tram_dist_gt[3] += 1
            elif line_tmp[1] == "cyclist":
                if float(line_tmp[2]) <= short_dist:
                    count_cyclist_dist_gt[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_cyclist_dist_gt[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_cyclist_dist_gt[2] += 1
                else:
                    count_cyclist_dist_gt[3] += 1
            elif line_tmp[1] == "misc":
                if float(line_tmp[2]) <= short_dist:
                    count_misc_dist_gt[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_misc_dist_gt[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_misc_dist_gt[2] += 1
                else:
                    count_misc_dist_gt[3] += 1
            elif line_tmp[1] == "person_sitting":
                if float(line_tmp[2]) <= short_dist:
                    count_person_sitting_dist_gt[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_person_sitting_dist_gt[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_person_sitting_dist_gt[2] += 1
                else:
                    count_person_sitting_dist_gt[3] += 1
            else:
                print("no such object")

            count_gt += 1

        print("count", count_gt)
        print("count_car_dist_gt", count_car_dist_gt)
        print("count_truck_dist_gt", count_truck_dist_gt)
        print("count_van_dist_gt", count_van_dist_gt)
        print("count_pedestrian_dist_gt", count_pedestrian_dist_gt)
        print("count_cyclist_dist_gt", count_cyclist_dist_gt)
        print("count_misc_dist_gt", count_misc_dist_gt)
        print("count_person_sitting_dist_gt", count_person_sitting_dist_gt)

print("=========================================================")
file_path = "./val_output_2019-09-13-14-15-09.txt"
count = 0
if os.path.exists(file_path):
    with open(file_path) as file:
        for line_num, line_ in enumerate(file):
            line_ = line_[:-1].split(' ')
            line_tmp = line_[1:]

            if line_tmp[1] == "car":
                if float(line_tmp[2]) <= short_dist:
                    count_car_dist[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_car_dist[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_car_dist[2] += 1
                else:
                    count_car_dist[3] += 1
            elif line_tmp[1] == "van":
                if float(line_tmp[2]) <= short_dist:
                    count_van_dist[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_van_dist[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_van_dist[2] += 1
                else:
                    count_van_dist[3] += 1
            elif line_tmp[1] == "truck":
                if float(line_tmp[2]) <= short_dist:
                    count_truck_dist[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_truck_dist[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_truck_dist[2] += 1
                else:
                    count_truck_dist[3] += 1
            elif line_tmp[1] == "pedestrian":
                if float(line_tmp[2]) <= short_dist:
                    count_pedestrian_dist[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_pedestrian_dist[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_pedestrian_dist[2] += 1
                else:
                    count_pedestrian_dist[3] += 1
            elif line_tmp[1] == "tram":
                if float(line_tmp[2]) <= short_dist:
                    count_tram_dist[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_tram_dist[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_tram_dist[2] += 1
                else:
                    count_tram_dist[3] += 1
            elif line_tmp[1] == "cyclist":
                if float(line_tmp[2]) <= short_dist:
                    count_cyclist_dist[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_cyclist_dist[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_cyclist_dist[2] += 1
                else:
                    count_cyclist_dist[3] += 1
            elif line_tmp[1] == "misc":
                if float(line_tmp[2]) <= short_dist:
                    count_misc_dist[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_misc_dist[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_misc_dist[2] += 1
                else:
                    count_misc_dist[3] += 1
            elif line_tmp[1] == "person_sitting":
                if float(line_tmp[2]) <= short_dist:
                    count_person_sitting_dist[0] += 1
                elif float(line_tmp[2]) <= mid_dist_1 and float(line_tmp[2]) > short_dist:
                    count_person_sitting_dist[1] += 1
                elif float(line_tmp[2]) <= mid_dist_2 and float(line_tmp[2]) > mid_dist_1:
                    count_person_sitting_dist[2] += 1
                else:
                    count_person_sitting_dist[3] += 1
            else:
                print("no such object")

            count += 1

        print("count", count)
        print("count_car_dist", count_car_dist)
        print("count_truck_dist", count_truck_dist)
        print("count_van_dist", count_van_dist)
        print("count_pedestrian_dist", count_pedestrian_dist)
        print("count_cyclist_dist", count_cyclist_dist)
        print("count_misc_dist", count_misc_dist)
        print("count_person_sitting_dist", count_person_sitting_dist)

print("****************************************************")
print("Show dist_stat", "short<=10", "20>=mid_1>10", "30>=mid_1>20", "long>30")
car_dist_stat = np.zeros(4)
for i in range(4):
    car_dist_stat[i] = count_car_dist[i] / count_car_dist_gt[i]
print("car_dist_stat", car_dist_stat)

truck_dist_stat = np.zeros(4)
for i in range(4):
    truck_dist_stat[i] = count_truck_dist[i] / count_truck_dist_gt[i]
print("truck_dist_stat", truck_dist_stat)

van_dist_stat = np.zeros(4)
for i in range(4):
    van_dist_stat[i] = count_van_dist[i] / count_van_dist_gt[i]
print("van_dist_stat", van_dist_stat)

cyclist_dist_stat = np.zeros(4)
for i in range(4):
    cyclist_dist_stat[i] = count_cyclist_dist[i] / count_cyclist_dist_gt[i]
print("cyclist_dist_stat", cyclist_dist_stat)

tram_dist_stat = np.zeros(4)
for i in range(4):
    tram_dist_stat[i] = count_tram_dist[i] / count_tram_dist_gt[i]
print("tram_dist_stat", tram_dist_stat)

pedestrian_dist_stat = np.zeros(4)
for i in range(4):
    pedestrian_dist_stat[i] = count_pedestrian_dist[i] / count_pedestrian_dist_gt[i]
print("pedestrian_dist_stat", pedestrian_dist_stat)

misc_dist_stat = np.zeros(4)
for i in range(4):
    misc_dist_stat[i] = count_misc_dist[i] / count_misc_dist_gt[i]
print("misc_dist_stat", misc_dist_stat)

person_sitting_dist_stat = np.zeros(3)
for i in range(3):
    person_sitting_dist_stat[i] = count_person_sitting_dist[i] / count_person_sitting_dist_gt[i]
print("person_sitting_dist_stat", person_sitting_dist_stat)