import numpy as np

arr = [[2, 5], [1, 3], [0, 9]]

nparr = np.array(arr)
print(nparr)

avg = np.average(nparr, axis = 0)
print(avg)
