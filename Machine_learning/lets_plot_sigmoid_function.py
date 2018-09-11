import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as  plt

x=np.linspace(-10,10,num=1000 )
plt.figure(figsize=(12,8))
sig=1/(1+np.exp(-x))#sigmoid=1/(1+e^-x))
plt.plot(x,sig)
plt.title('Sigmoid funtion')
plt.show()
