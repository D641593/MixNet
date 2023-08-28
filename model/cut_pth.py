# cut the embed head in current pth file


import os
import torch

path = "ArT_mid/MixNet_FSNet_M_160.pth"

cpt = torch.load(path)
model_cpt = cpt["model"]
print(model_cpt.keys())

keys_list = list(model_cpt.keys())
for key in keys_list:
    if "embed_" in key:
        model_cpt.pop(key)

print(model_cpt.keys())
cpt["model"] = model_cpt
torch.save(cpt, "MixNet_FSNet_M_160.pth")