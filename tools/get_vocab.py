import sys
import json
import yaml
import collections

infile=open(sys.argv[1])
outfile=open(sys.argv[2], "w")

## default: {'<pad>': 0, '<unk>': 1, '<s>': 3}
dict_tmp = {'<pad>': 0}
yaml.dump(dict_tmp, outfile)
dict_tmp = {'<unk>': 1}
yaml.dump(dict_tmp, outfile)
dict_tmp = {'<s>': 2}
yaml.dump(dict_tmp, outfile)

i = 3
dict_tmp = {}
for line in infile:
    items = line.strip().split(" ")
    dict_tmp[items[0]] = i
    yaml.dump(dict_tmp, outfile)
    dict_tmp = {}
    i += 1

infile.close()
outfile.close()
