#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:47:34 2018

"""

import json
import pandas as pd
import os

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
classes=['bus','light','traffic_sign','person','bike','truck','motor','car','train','Rider']

filename1='/home/mayank-s/PycharmProjects/Datasets/Berkely DeepDrive/labels/100k/train/'
for i in os.listdir(filename1):

    data=open(filename1+i,'r')
    data1 = data.read()
    data.close()
    Json = json.loads(data1)
    filename.append(Json['name'])
    for ki in Json['frames'][0]['objects']:
        print(ki)
        print(ki.box2d.category)


    print(len(Json['frames'][0]['objects']))
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
    file_number=file_number+1
columns=['filename','Class','xmin','xmax','ymin','ymax']
pd1=pd.DataFrame(a,columns=columns)
pd1.to_csv('output.csv')

