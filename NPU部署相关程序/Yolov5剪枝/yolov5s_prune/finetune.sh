#python train.py --img 640 --batch 32 --epochs 100 --weights runs/val/exp2/pruned_model.pt  --data data/coco_hand.yaml --cfg models/yolov5s.yaml --name coco_hand_ft --device 0 --optimizer AdamW --ft_pruned_model --hyp hyp.finetune_prune.yaml

# 213服务器上微调、再训练
python -m torch.distributed.launch --nproc_per_node 2 train.py --img 1024 --batch 64 --epochs 100 --weights runs/val/exp8/pruned_model.pt  --data data/PrivacyMaskingCarPlateFace.yaml --cfg models/yolov5s.yaml --name carplate_face --device 2,3 --optimizer AdamW --ft_pruned_model --hyp hyp.finetune_prune.yaml