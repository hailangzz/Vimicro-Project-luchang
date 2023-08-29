# python prune.py --percent 0 --weights runs/train/coco_hand_sparsity6/weights/last.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --imgsz 640
# python prune.py --percent 0.5 --weights runs/train/coco_sparsity2/weights/last.pt --data data/coco.yaml --cfg models/yolov5s.yaml --imgsz 640

# 213服务器上剪枝：
python prune.py --percent 0.43 --weights D:\迅雷下载\AI数据集汇总\\runs/train/PrivacyMasking_yolov5s_pruned3/weights/last.pt --data  data/PrivacyMaskingCarPlateFace.yaml --cfg models/yolov5s_PrivacyMaskingCarPlateFace.yaml

python prune.py --percent 0.43 --weights ./runs/train/PrivacyMasking_yolov5s_pruned3/weights/last.pt --data  data/PrivacyMaskingCarPlateFace.yaml --cfg models/yolov5s_PrivacyMaskingCarPlateFace.yaml

