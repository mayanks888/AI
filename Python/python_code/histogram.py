import matplotlib.pyplot as plt
import numpy as np

x = np.random.random_integers(1, 100, 5)
print(x)
plt.axis([0, 100, 0, 2])
plt.hist(x, bins=20)
# understanding bin is important : bin means dividing the range into bin
# for eg: if no of bin are 2 then divide the range into 2 ranges o.e( range by 2)
# for 10 : divide into 10 equal space
plt.ylabel('No of times')
plt.show()
