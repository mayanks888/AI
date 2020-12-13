import matplotlib.pyplot as plt

# tensor([0.0485, 0.4602, 0.7500, 0.0000, 0.0000, 0.0000, 0.0000])
# tensor([0.5779, 0.1977, 0.0342, 0.0000, 0.0000, 0.0000, 0.0000])


#        [0.1,     0.3,       0.5,    0.7, 0.8, 0.9, 0.95]
prec = [0.0462578, 0.6166667, 0.8333333, 0, 0, 0, 0]
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
prec = [0.1484906822, 0.7121905088, 0.9558198452, 0.9964547753, 0.9990401864, 0.999938786, 1]
rec = [0.8950397968, 0.628179729, 0.3427012265, 0.1406136006, 0.0684016049, 0.0173126496, 0.0048565301]

fig, ax = plt.subplots()
ax.plot(rec, prec)

for i, txt in enumerate(threshold):
    ax.annotate(txt, (rec[i], prec[i]))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# prec=[.05,.53,.90,0.99,1,0,0]
# rec=[0.54,0.20,.053,.0073,.000002,0,0]
#  pr.prec@10=0.05027, pr.rec@10=0.5453, pr.prec@30=0.5398, pr.rec@30=0.2023, pr.prec@50=0.9074, pr.rec@50=0.05385, pr.prec@70=0.9939, pr.rec@70=0.0007315, pr.prec@80=1.0, pr.rec@80=2.24e-06, pr.prec@90=0.0, pr.rec@90=0.0, pr.prec@95=0.0, pr.rec@95=0.0, misc.num_vox=87055, misc.num_pos=240, misc.num_neg=307074, misc.num_anchors=307520, misc.lr=0.001554, misc.mem_usage=22.8
