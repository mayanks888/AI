import matplotlib.pyplot as plt

# tensor([0.0485, 0.4602, 0.7500, 0.0000, 0.0000, 0.0000, 0.0000])
# tensor([0.5779, 0.1977, 0.0342, 0.0000, 0.0000, 0.0000, 0.0000])


#        [0.1,     0.3,       0.5,    0.7, 0.8, 0.9, 0.95]
prec1 = [0.0462578, 0.6166667, 0.8333333, 0, 0, 0, 0]
rec = [0.872549, 0.3627451, 0.14705883, 0, 0, 0, 0]
threshold = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
# plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plt.scatter(prec,rec)
# plt.plot(rec,prec)
# # plt.plot(fpr, tpr, marker='.')
# # show the plot
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.show()


prec = [.05, .53, .90, 0.99, 1, 0, 0]
rec = [0.54, 0.20, .053, .0073, .000002, 0, 0]

fig, ax = plt.subplots()
ax.plot(rec, prec)

for i, txt in enumerate(threshold):
    ax.annotate(txt, (rec[i], prec[i]))

plt.xlabel('Recall')
plt.ylabel('Precision')
# plt.show()

# prec=[.05,.53,.90,0.99,1,0,0]
# rec=[0.54,0.20,.053,.0073,.000002,0,0]
#  pr.prec@10=0.05027, pr.rec@10=0.5453, pr.prec@30=0.5398, pr.rec@30=0.2023, pr.prec@50=0.9074, pr.rec@50=0.05385, pr.prec@70=0.9939, pr.rec@70=0.0007315, pr.prec@80=1.0, pr.rec@80=2.24e-06, pr.prec@90=0.0, pr.rec@90=0.0, pr.prec@95=0.0, pr.rec@95=0.0, misc.num_vox=87055, misc.num_pos=240, misc.num_neg=307074, misc.num_anchors=307520, misc.lr=0.001554, misc.mem_usage=22.8


import pandas as pd

data = pd.read_csv('/home/mayank_sati/Desktop/pp/pr/new.csv')

print(1)
print(data.iloc[:, 2:8])

time = data.iloc[:, 1].values
prec = data.iloc[:, 2:9].values
rec = data.iloc[:, 9:18].values

# prec=prec.tolist()
# rec=rec.tolist()
# for iIndex, box3d in enumerate(boxes_lidar, start=0):
for iIndex, t in enumerate(time, start=0):
    print(t)
    if t > 300:
        prec_val = prec[iIndex]
        rec_val = rec[iIndex].tolist()
        prec_val = prec_val.tolist()
        fig, ax = plt.subplots()
        ax.plot(rec_val, prec_val)

        for i, txt in enumerate(threshold):
            ax.annotate(txt, (rec_val[i], prec_val[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.show()
        path = "/home/mayank_sati/Desktop/pp/pr/curve"
        path = path + "/" + str(t) + ".jpg"
        plt.savefig(path)

# z= dataset.iloc[:, 3].values
# for loop in x:
#     print(loop)
# # [Xmin,ymin,xmax,ymax]
#     top = (y[val,0], y[val,3])
#     bottom = (y[val,2], y[val,1])

# mydata=data.groupby('Step').values
# print(data.groupby('Step').count())
# # index=mydata.groups['car'].values
# #precisiion vs recal at diff threshold curve
# for dat in mydata:
#     print(dat)
#     prec=[dat[1],dat[2],dat[3],dat[4],dat[5],dat[6],dat[7],dat[7],dat[8],dat[9],dat[10],dat[11],dat[12],dat[13],dat[14],dat[15]]


# plt.figure(figsize=(12,8));
# plt.plot(precisions, recalls);
# plt.xlabel('recalls');
# plt.ylabel('precisions');
# plt.title('PR Curve: precisions/recalls tradeoff');
# plt.show()
