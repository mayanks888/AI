import time
from datetime import datetime
# numpy_conv_pfn = features.cpu().detach().numpy()
ts = time.time()
st = datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S_%f')
print(st)