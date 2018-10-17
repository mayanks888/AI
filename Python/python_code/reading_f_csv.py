import pandas as pd
import numpy as np
import re
data=pd.read_csv("f.csv")
print(data.head())
new=data["Irms(Amps)"]
focal_val=np.array(new)
print(focal_val)

data="b'3.14$1.21$5.56a7.76\r\n'"

# # cool=re.findall(r'\b\d+\b', data)
# cool=re.findall(r'\d', data)
cool=re.findall("\d+\.\d+", data)

print(cool)
print(cool[0])
print(cool[1])