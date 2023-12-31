# python train.py --batch 32 --epochs 100 --weights runs/train/coco_hand/weights/last.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --name coco_hand_sparsity --optimizer AdamW --bn_sparsity --sparsity_rate 0.0001 --device 3
# python train.py --batch 32 --epochs 50 --weights weights/yolov5s.pt --data data/coco.yaml --cfg models/yolov5s.yaml --name coco_sparsity --optimizer AdamW --bn_sparsity --sparsity_rate 0.0001 --device 3 --hyp data/hyps/hyp.finetune.yaml

#python train.py --batch 32 --epochs 50 --weights weights/yolov5s.pt --data data/coco.yaml --cfg models/yolov5s.yaml --name coco_sparsity --optimizer AdamW --bn_sparsity --sparsity_rate 0.00005 --device 3

# 213服务器上训练：
python  train.py --batch 64 --epochs 100 --weights pretrain_model/yolov5s.pt --data data/PrivacyMaskingCarPlateFace.yaml --cfg models/yolov5s_PrivacyMaskingCarPlateFace.yaml --name ./runs/train/ --name PrivacyMasking_yolov5s_pruned  --optimizer AdamW --bn_sparsity --sparsity_rate 0.00005 --device 2,3











