import sys

import torch

if len(sys.argv) > 1:
    file = sys.argv[1] or 'latest.pth'
else:
    file = 'latest.pth'
print()
cp = torch.load(file)
if 'state_dict' in cp.keys():
    print(cp.keys())
    sd = cp['state_dict']
else:
    sd = cp
print()
dct = {}
for k, v in sd.items():
    m = k.split('.')[0]
    if m in dct.keys():
        dct[m] = dct[m] + v.numel()
    else:
        dct[m] = v.numel()

if len(dct) <= 1:
    dct = {}
    for k, v in sd.items():
        m = '.'.join(k.split('.')[:2])
        if m in dct.keys():
            dct[m] = dct[m] + v.numel()
        else:
            dct[m] = v.numel()

for k, v in dct.items():
    print(k + ":\t", v)
print('-' * 26)
print("tot:\t\t", sum(dct.values()))
