# Fully Convolutional One-Stage (FCOS) Object Detector with ConvNeXt-Tiny

Mô hình phát hiện đối tượng anchor-free FCOS tự xây dựng từ đầu (from scratch) sử dụng mạng trích xuất đặc trưng ConvNeXt-Tiny và mạng kim tự tháp đặc trưng (FPN). Dự án hỗ trợ huấn luyện trên GPU T4 và suy luận hiệu quả.

---

## 1. Cấu trúc thư mục

```
my_submission/
├── models/
│   ├── backbone.py        # Mạng trích xuất đặc trưng ConvNeXt-Tiny
│   ├── fpn.py             # Feature Pyramid Network (FPN)
│   ├── fcos_head.py       # Đầu dự đoán classification, regression, centerness
│   └── fcos.py            # Kết hợp Backbone + FPN + Head
├── utils/
│   ├── dataset.py         # Bộ đọc dữ liệu + augmentation (Letterbox, Mosaic, Flip, Jitter)
│   ├── assign.py          # Thuật toán gán nhãn pixel-wise (FCOS Assignment)
│   ├── loss.py            # Tổ hợp Loss: Focal Loss + CIoU Loss + BCE Centerness
│   └── nms.py             # Khử trùng hộp bao Soft-NMS (Class-wise)
├── train.py              # Script huấn luyện
├── predict.py            # Script suy luận
├── requirements.txt      # Thư viện yêu cầu
└── README.md             # Hướng dẫn sử dụng này
```

---

## 2. Cài đặt môi trường

Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```
Nếu huấn luyện trên GPU (khuyên dùng), hãy đảm bảo cài đặt đúng phiên bản PyTorch hỗ trợ CUDA.

---

## 3. Hướng dẫn huấn luyện (Training)

Lệnh huấn luyện bắt buộc để chạy huấn luyện trên bộ dữ liệu và lưu checkpoint tốt nhất vào thư mục `./models/best.pth`:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 35 \
  --batch_size 16 \
  --early_stop_patience 10 \
  --min_epochs 25
```

### Các chiến lược tối ưu trong quá trình huấn luyện:
- **Warmup**: Đóng băng backbone ở 3 epoch đầu và tăng tuyến tính learning rate head lên `1e-3`.
- **OneCycleLR**: Dùng cho epoch 3-27 với LR backbone bằng `0.1 * LR head` để rút ngắn hội tụ.
- **EMA (Exponential Moving Average)**: Cập nhật trọng số trung bình động của mô hình giúp nâng cao độ chính xác kiểm thử thêm từ 0.5% - 1.5% mAP.
- **EMA warmup**: Decay được tăng dần ở giai đoạn đầu để validation không bị kẹt ở trọng số khởi tạo.
- **Unfreeze backbone**: Mở toàn bộ backbone từ epoch 3, sau đó freeze lại ở epoch 28 để fine-tune head ổn định hơn.
- **Plateau switch**: Nếu mAP tăng dưới `0.005` trong 3 validation liên tiếp sau epoch 13, training tự chuyển fine-tune sớm.
- **35-epoch schedule**: 3 epoch warmup, 25 epoch main training, 7 epoch fine-tune.
- **Early stopping**: Dừng huấn luyện nếu mAP validation không cải thiện sau một số epoch nhất định, mặc định sau tối thiểu 25 epoch và patience 10.

---

## 4. Hướng dẫn suy luận (Inference/Prediction)

Lệnh suy luận bắt buộc để xuất kết quả dự đoán ra tệp JSON:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint ./models/best.pth \
  --conf_thresh 0.05
```

Kết quả dự đoán sẽ được lưu dưới dạng file `predictions.json` theo đúng cấu trúc yêu cầu của đề bài:
```json
[
  {
    "image_id": "img_name.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [xmin, ymin, xmax, ymax]
      }
    ]
  }
]
```

---

## 5. Vị trí đặt mô hình và trọng số
Trọng số tốt nhất được huấn luyện và lưu tự động vào:
- `./models/best.pth` (lưu trọng số của mô hình EMA tốt nhất).
- `./models/last.pth` (lưu checkpoint đầy đủ bao gồm epoch, optimizer và EMA weights để tiếp tục huấn luyện nếu bị gián đoạn).
