# MixNet
This is the official code for MixNet: Toward Accurate Detection of Challenging Scene Text in the Wild

# docker environment
[Click Here](<https://drive.google.com/file/d/1qd7M6Zh3l0XEHAFugK_WAPMbcNe5iPMY/view?usp=sharing>)
# Evaluation Result on Benchmark 
|Datasets | Prec. (%)| Recall (%) | F1-score (%) | weight (.pth)
|-----|--------|--------------|----------|-------------------|
|Total-Text|93.0|88.1|90.5|[model](<https://drive.google.com/file/d/1t2LDzXsIBDIS3DAPcR5hpuOAzcN_sznB/view?usp=sharing>)|
|MSRA-TD500|90.7|88.1|89.4|[model](<https://drive.google.com/file/d/1xba77OIDASXJYEUWDfgIPWXNG34rHN-S/view?usp=sharing>)|
|ICDAR-ArT|83.0|76.7|79.7|[model](<https://drive.google.com/file/d/17SOTd34cLBmZCtQDrtlfBjxhbQgkA5GS/view?usp=sharing>)|
|CTW1500  |91.4|88.3|89.8|[model](<https://drive.google.com/file/d/1QTlAYQuCBQKM-0CcXNkDKj2vJ2Hn3MZH/view?usp=sharing>)|

# Evaluation Result on CTW1500
This section elucidates the performance evaluation on the CTW1500 dataset. 

When utilizing the [TIoU-metric-python3](<https://github.com/PkuDavidGuan/TIoU-metric-python3>) scoring code, our model's scores are as presented below:
|Datasets | Prec. (%)| Recall (%) | F1-score (%) |
|-----|--------|--------------|----------|
|CTW1500  |90.3|84.8|87.5|

However, upon inputting MixNet's output into the [DPText-DETR](<https://github.com/ymy-k/DPText-DETR>)'s calculation program, the ensuing results differ:
|Datasets | Prec. (%)| Recall (%) | F1-score (%) |
|-----|--------|--------------|----------|
|CTW1500  |91.4|88.3|89.8|

I'm not sure why the data is inconsistent. Therefore, I've provided the scores obtained from both calculations for reference.

# Eval
```bash
  # Total-Text
  python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name Totaltext_mid --checkepoch 622 --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --mid True
  # CTW1500
  python3 eval_mixNet.py --net FSNet_hor --scale 1 --exp_name Ctw1500 --checkepoch 925 --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85
  # MSRA-TD500
  python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name TD500HUST_mid --checkepoch 284 --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --mid True
  # ArT
  python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name ArT_mid --checkepoch 160 --test_size 960 2880 --dis_threshold 0.4 --cls_threshold 0.8 --mid True
```
# Acknowledgement
This code has been modified based on the foundation laid by [TextBPN++](<https://github.com/GXYM/TextBPN-Plus-Plus>). <br>
We use code from [Connected_components_PyTorch](<https://github.com/zsef123/Connected_components_PyTorch>) as post-processing. <br> 
Thanks for their great work! <br>
