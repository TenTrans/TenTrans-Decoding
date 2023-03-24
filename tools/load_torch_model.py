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
    

checkpoint = torch.load(sys.argv[1])
convert_npz = sys.argv[2]
isFp16 = sys.argv[3]

model_sentenceRep = checkpoint["model_sentenceRep"]
model_target = checkpoint["model_target"]
states_dict = {}


# 1. extract params from encoder
for k in model_sentenceRep:
    v = model_sentenceRep[k]
    print(k, v.size())
    if (k.startswith("encoder.layers")) and (v.dim() == 2):
        if isFp16:
            states_dict[k] = v.t().clone().detach().cpu().numpy().astype(np.float16)
        else:
            states_dict[k] = v.t().clone().detach().cpu().numpy()
        print("[TransPose]...\n")
        print(">>>>>>>>>>>>>>>>> before <<<<<<<<<<<<<<<<<<<<")
        print(v)
        print(">>>>>>>>>>>>>>>>> after <<<<<<<<<<<<<<<<<<<<")
        print(states_dict[k].shape, states_dict[k])
        print("\n\n")
    else:
        if isFp16:
            states_dict[k] = v.detach().cpu().numpy().astype(np.float16)
        else:
            states_dict[k] = v.detach().cpu().numpy()

# 2. extract params from decoder
for k in model_target:
    v = model_target[k]
    print(k, v.size())
    if (k.startswith("decoder.layers") or k.startswith("decoder.output_layer")) and (v.dim() == 2):
        if isFp16:
            states_dict[k] = v.t().clone().detach().cpu().numpy().astype(np.float16)
        else:
            states_dict[k] = v.t().clone().detach().cpu().numpy()
        print("[TransPose]...\n")
        print(">>>>>>>>>>>>>>>>> before <<<<<<<<<<<<<<<<<<<<")
        print(v)
        print(">>>>>>>>>>>>>>>>> after <<<<<<<<<<<<<<<<<<<<")
        print(states_dict[k].shape, states_dict[k])
        print("\n\n")
    else:
        if isFp16:
            states_dict[k] = v.detach().cpu().numpy().astype(np.float16)
        else:
            states_dict[k] = v.detach().cpu().numpy()

# 3. remove redundant params
if checkpoint['config']['target']['share_all_embedd']:
    del states_dict["decoder.embedding.weight"]
if checkpoint['config']['target']['share_out_embedd']:
    del states_dict["decoder.output_layer.weight"]

# 4. save as .npz model
savez(convert_npz, states_dict)


print(">>>>>>>>>>>>>>>>> npz model test <<<<<<<<<<<<<<<<<<<<")
model = np.load(convert_npz)
for k in model:
    v = model[k]
    print(k, v.shape)
    print(v)