import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,5,10)
y=x**2
y1=x**3
'''print (x)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel('y')
plt.title('my plot')
plt.show()

#creating subplot
# this is a interesting way to plot multiplot in one windows
plt.subplot(1,2,1)#1=rows,2=colum,1=position
plt.plot(x,y,'b')
plt.subplot(1,2,2)
plt.plot(x,y1,'r')
plt.show()

#create you own figure with axis
#interesting play with it
# plt.plot(x,y,'b')

fig=plt.figure()
myaxis=fig.add_axes([.1,.1,.8,.8])#add axis function is function of shofting your plot in a wholw page(left right up down)
# A 4-length sequence of [left, bottom, width, height] quantities.
myaxis2=fig.add_axes([.2,.5,.3,.3])
myaxis.plot(x,y)
myaxis2.plot(y,x)
plt.show()

fig,axes=plt.subplots(nrows=1,ncols=2)
axes[0]=plt.plot(x,y)
axes[1]=plt.plot(y1,x)
plt.tight_layout()#set a plot properly into page
plt.show()
plt.savefig('myfigure.png')#saving a plot'''

fig=plt.figure()
ax=fig.add_axes([.1,.1,1,1])
ax.plot(x,y,color='purple',linewidth=3)
ax.set_xlim([0,5])#define the axis value as your convinient
ax.set_ylim([0,10])
plt.show()
