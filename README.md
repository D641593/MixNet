# MixNet
This is the official code for MixNet: Toward Accurate Detection of Challenging Scene Text in the Wild
|Datasets | Prec. (%)| Recall (%) | F1-score |
|-----|--------|--------------|----------|
|Total-Text|93.0|88.1|90.5|
|CTW1500  |91.4|88.3|89.8|
|MSRA-TD500|90.7|88.1|89.4|
|ICDAR-ArT|83.0|76.7|79.7|


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
