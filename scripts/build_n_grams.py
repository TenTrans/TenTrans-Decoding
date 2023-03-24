import sys
import torch
import numpy as np
import format
from numpy.compat import (asbytes, asstr, asunicode, os_fspath, os_PathLike, 
        pickle, contextlib_nullcontext)

def zipfile_factory(file, *args, **kwargs):
    """
    Create a ZipFile.
    Allows for Zip64, and the `file` argument can accept file, str, or
    pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile
    constructor.
    """
    if not hasattr(file, 'read'):
        file = os_fspath(file)
    import zipfile
    kwargs['allowZip64'] = True
    return zipfile.ZipFile(file, *args, **kwargs)

def savez(infile, args, compress=False, allow_pickle=True, pickle_kwargs=None):
    import zipfile
    
    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    zipf = zipfile_factory(infile, mode='w', compression=compression)
    for key, val in args.items():
        fname = key + '.npy'
        val = np.asanyarray(val)
        # always force zip64, gh-10776
        with zipf.open(fname, 'w', force_zip64=True) as fid:
            format.write_array(fid, val, allow_pickle=allow_pickle, pickle_kwargs=pickle_kwargs)

    zipf.close()

align=open(sys.argv[1], "r")
data=open(sys.argv[2], "r")
convert_npz = sys.argv[3]

n_dict = {}

## x: 0-1 1-2 ...
## y: aa a ||| aa a
for x, y in zip(align.readlines(), data.readlines()):
    items = y.strip().split(" ||| ")
    src = items[0].strip().split(" ")
    tgt = items[1].strip().split(" ")

    items = x.strip().split(" ")
    indexs = []
    for item in items:
        tmp = item.strip().split("-")
        i, j = int(tmp[0]), int(tmp[1])

        cur_src, cur_tgt = src[i], tgt[j]
        if n_dict.get(cur_src) is None:
            n_dict[cur_src] = {}
            n_dict[cur_src][cur_tgt] = 1
        else:
            if n_dict[cur_src].get(cur_tgt) is None:
                n_dict[cur_src][cur_tgt] = 1
            else:
                n_dict[cur_src][cur_tgt] += 1

savez(convert_npz, n_dict)
