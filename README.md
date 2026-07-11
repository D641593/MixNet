# MixNet – Custom Dataset Setup Notes

Tổng hợp toàn bộ phần đã chỉnh/vá để chạy được **MixNet**
(https://github.com/D641593/MixNet) trên môi trường hiện đại (NumPy/Shapely/PyTorch mới)
và train/infer trên dataset riêng (định dạng nhãn kiểu PaddleOCR).

Repo gốc có khá nhiều chỗ không chạy được "as-is" trên môi trường mới — tất cả các
file `.py` trong bộ này đều là **file vá/thêm vào repo gốc**, không phải viết lại từ đầu.

---

## 1. Danh sách file

| File | Đặt vào đâu | Vai trò |
|---|---|---|
| `apply_compat_patches.py` | gốc repo (`MixNet/`) | Vá tương thích NumPy/Shapely/PyTorch mới, bỏ dependency `mmcv` |
| `Custom_Text_mid.py` | `MixNet/dataset/` | Loader đọc trực tiếp label PaddleOCR (`train_det.txt`/`val_det.txt`) |
| `option.py` | `MixNet/cfglib/` (đè file cũ) | Thêm choice `Custom_mid`, đổi mặc định `--val_freq` |
| `train_mixNet.py` | gốc repo (đè file cũ) | Thêm nhánh dataset `Custom_mid`, hàm `validate()` chạy P/R/F1 định kỳ, sửa bug không save checkpoint |
| `infer_single_image.py` | gốc repo (`MixNet/`) | Chạy detect trên 1 ảnh, không cần dataset loader |

---

## 2. Cài đặt

### 2.1. Copy file vào repo
```bash
cd MixNet
cp /path/to/apply_compat_patches.py .
cp /path/to/Custom_Text_mid.py dataset/
cp /path/to/option.py cfglib/option.py
cp /path/to/train_mixNet.py train_mixNet.py
cp /path/to/infer_single_image.py .
```

### 2.2. Vá tương thích thư viện (chạy 1 lần, idempotent)
```bash
python3 apply_compat_patches.py
```
Script này tự vá:
- `np.float` / `np.int` / `np.bool` / `np.int0` — bị NumPy ≥1.24 xoá hẳn, dùng khắp `dataset/`, `util/`, `network/`.
- `np.array(shapely_geometry)` trong `dataset/dataload_midline.py` — vỡ trên Shapely ≥2.0, dùng để build midline supervision.
- `np.reshape(..., newshape=...)` trong `network/Reg_loss.py` — NumPy mới đổi tên keyword thành `shape=`.
- `config.pos` / `config.resume` — `network/textnet.py` đọc 2 field này nhưng repo gốc không hề set default, gây `AttributeError` khi build model (train/eval/infer đều dính).
- Bỏ dependency `mmcv` khỏi `dataset/TD500HUST_Text.py` (thay `mmcv.list_from_file` bằng hàm đọc file 3 dòng) — `mmcv` chỉ dùng đúng 1 hàm này, mà cài `mmcv` trên môi trường mới hay lỗi `ModuleNotFoundError: No module named 'pkg_resources'`.

### 2.3. Cài dependency còn thiếu (không có trong `requirements.txt`)
```bash
pip install pycocotools ujson pytorch-ssim tensorboardX --break-system-packages
```
Lý do: `import dataset` sẽ nạp **toàn bộ** loader trong `dataset/__init__.py` kể cả loader bạn không dùng, nên cần đủ mấy gói này thì package mới import được.

---

## 3. Train trên dataset riêng (định dạng PaddleOCR)

### 3.1. Format dữ liệu

`Custom_Text_mid.py` đọc trực tiếp label kiểu PaddleOCR/PPOCRLabel export, **không cần convert**:

```
data/receipt/
├── imgs/...                  # đúng path tương đối ghi trong 2 file txt bên dưới
├── train_det.txt
└── val_det.txt                # optional, chỉ cần nếu muốn validate/eval
```

Mỗi dòng (tab-separated):
```
imgs/10122_5.jpg\t[{"transcription": "", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]
```

### 3.2. Wiring (đã có sẵn trong file `train_mixNet.py`/`option.py`/`Custom_Text_mid.py` đính kèm)

Cần tự thêm 2 chỗ vào `dataset/__init__.py`:
```python
from .Custom_Text_mid import CustomText_mid
```

Và vào `eval_mixNet.py` (nếu muốn eval riêng bằng script này, xem mục 5):
```python
from dataset.Custom_Text_mid import CustomText_mid
```
+ nhánh trong `main()`:
```python
elif cfg.exp_name == "Custom_mid":
    testset = CustomText_mid(
        data_root='data/receipt',
        is_training=False,
        transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
    )
```

### 3.3. Chạy train
```bash
python3 train_mixNet.py \
    --exp_name Custom_mid --net FSNet_M --mid True \
    --resume model/Totaltext_mid/MixNet_FSNet_M_622.pth \
    --scale 1 --input_size 640 \
    --batch_size 8 --max_epoch 300 \
    --optim Adam --lr 0.0001 \
    --num_workers 8 --save_freq 5 --val_freq 5
```

**Lưu ý bắt buộc:** luôn truyền `--resume <checkpoint .pth có sẵn>`. Nếu không, `network/layers/FSNet.py` sẽ tự động tìm load
`./pretrained/triHRnet_Synth_weight.pth` — file này **không có trong repo, không có link tải** — sẽ crash ngay khi build model.
Dùng 1 trong các checkpoint chính thức (`Totaltext_mid`, `TD500HUST_mid`, `ArT_mid`, `Ctw1500`) tải từ README gốc để fine-tune, vừa né crash vừa tốt hơn train from-scratch trên dataset nhỏ.

### 3.4. Log khi chạy đúng
```
[CustomText_mid] train_det.txt: 897 samples loaded, 0 lines skipped
[CustomText_mid] val_det.txt: 100 samples loaded, 0 lines skipped
...
Missing key in state_dict, try to load with strict = False   <- bình thường, xem mục 4
Epoch: 0 : LR = [0.0001]
(0 / 113)  total_loss: 3.29  cls_loss: 0.46  ...
Saving to ./model/Custom_mid/MixNet_FSNet_M_5.pth.
Validation @ epoch 5: precision=0.xxxx recall=0.xxxx f1=0.xxxx
```
Checkpoint lưu tại `model/Custom_mid/MixNet_FSNet_M_<epoch>.pth`, mỗi `save_freq` epoch (mặc định 5, **không** còn bị khoá tới epoch 150 như code gốc). `validate()` chạy P/R/F1 (IoU@0.5, greedy matching — metric đơn giản để theo dõi xu hướng, không phải benchmark chính thức) mỗi `val_freq` epoch (mặc định 5).

---


## 4. Eval riêng (không bắt buộc, để có số liệu kỹ hơn)

```bash
python3 eval_mixNet.py \
    --exp_name Custom_mid --net FSNet_M --mid True \
    --checkepoch 50 \
    --test_size 640 1024 \
    --dis_threshold 0.3 --cls_threshold 0.85
```
(`--checkepoch 50` khớp tên file `model/Custom_mid/MixNet_FSNet_M_50.pth`.)

---

## 5. Detect trên 1 ảnh sau khi train xong

```bash
python3 infer_single_image.py \
    --image path/to/anh.jpg \
    --checkpoint model/Custom_mid/MixNet_FSNet_M_<epoch>.pth \
    --net FSNet_M --mid True \
    --test_size 640 1024 \
    --dis_threshold 0.3 --cls_threshold 0.85 \
    --output result.jpg --output_json result.json
```
Script này tự peek vào checkpoint để phát hiện có `embed_head` hay không rồi patch phù hợp — không cần chỉnh gì thêm dù dùng checkpoint tự train (không có `embed_head`) hay checkpoint gốc tải về (`Totaltext_mid`, có `embed_head`).