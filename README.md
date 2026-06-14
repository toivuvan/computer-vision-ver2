# YOLO-like ConvNeXt Detector

Mo hinh phat hien doi tuong anchor-free tu cai dat, dung ConvNeXt-Tiny pretrained backbone, YOLO-like PAN-FPN, decoupled head va Soft-NMS.

Ban hien tai duoc phat trien tu baseline `fa2d65ab update ver 5` da dat khoang `mAP@0.5 = 0.8035` tren validation. Thay doi moi: them centerness/quality branch that de giam false positive.

## Kien Truc

```
Input 416x416
  -> ConvNeXt-Tiny pretrained backbone
  -> YOLO-like PAN-FPN + SPPF + C2PSA-lite
  -> Decoupled head:
       - classification logits
       - l/t/r/b regression
       - centerness quality logits
  -> FCOS-style center sampling assignment
  -> Loss: Focal + CIoU + BCE centerness
  -> Soft-NMS
```

Inference score:

```text
score = sigmoid(class_logit) * sigmoid(centerness_logit)
```

Centerness target da co san trong `utils/assign.py`, nen thay doi nay tan dung tin hieu da ton tai thay vi de dummy zero nhu baseline cu.

## Train

Nen train lai vao folder moi vi checkpoint baseline cu thieu weights cua `centerness_preds`.

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models_centerness/ \
  --epochs 35 \
  --batch_size 16 \
  --img_size 416 \
  --early_stop_patience 10 \
  --min_epochs 25
```

Schedule hien tai:

- Epoch 0-2: freeze backbone, warmup head LR len `1e-3`.
- Epoch 3-27: unfreeze backbone, OneCycleLR, bat Mosaic/random scale.
- Fine-tune: freeze backbone, tat Mosaic/random scale, LR `5e-6`.
- EMA dung de evaluate va save `best.pth`.

## Inference

```bash
python predict.py \
  --image_dir ./public/test/images \
  --output predictions.json \
  --checkpoint ./models_centerness/best.pth \
  --img_size 416 \
  --conf_thresh 0.05
```

Chuyen JSON sang CSV:

```bash
python public/tools/convert.py \
  --input predictions.json \
  --output submission.csv \
  --classes ./public/classes.json
```

## Ghi Chu

- Checkpoint `models_yolo_like/best.pth` cua baseline 0.8035 se khong load strict vao model moi vi thieu `head.centerness_preds.*`.
- Neu muon so sanh cong bang, train lai tu dau va so voi `models_yolo_like/best.pth`.
- Neu centerness lam recall giam qua manh, thu giam nguong inference ve `--conf_thresh 0.03`.
