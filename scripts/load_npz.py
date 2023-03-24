import sys
import numpy as np

## infile=open(sys.argv[1], "r")

npz = sys.argv[1]
n_dict = np.load(npz, allow_pickle=True)
print(n_dict)

for key in n_dict.keys():
    print(key, n_dict[key])

print(n_dict['five-year'])
