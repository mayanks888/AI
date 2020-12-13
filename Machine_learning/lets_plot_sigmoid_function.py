import matplotlib.pyplot as  plt
import numpy as np

x = np.linspace(-10, 10, num=1000)
plt.figure(figsize=(12, 8))
sig = 1 / (1 + np.exp(-x))  # sigmoid=1/(1+e^-x))
print(sig)
plt.plot(x, sig)
plt.title('Sigmoid funtion')
plt.show()
