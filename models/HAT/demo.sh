# demo train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx2_from_scratch.yml --launcher pytorch

# 1
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 \
hat/train.py -opt options/train/My_train_HAT-L_SRx4_DIV2K_from_scratch_L1.yml --launcher pytorch

# 2
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 \
hat/train.py -opt options/train/My_train_HAT-L_SRx4_DF2K_from_scratch_L1.yml --launcher pytorch

# demo test
python hat/test.py -opt options/test/My_test_HAT-L_SRx4_DIV2K_from_scratch_L1.yml
python hat/test.py -opt options/test/My_test_HAT-L_SRx4_DF2K_from_scratch_L1.yml
python hat/test.py -opt options/test/My_test_HAT-L_SRx4_DF2K_from_scratch_L1_TTA.yml
python hat/test.py -opt options/test/My_test_HAT-L_SRx4_DF2K_from_scratch_L1_TTA_LR.yml

# 1
CUDA_VISIBLE_DEVICES=1 python hat/train.py -opt options/train/001train_denoise_HAT-S_SRx1_DIV2K_from_scratch_L2.yml

# inference
python infer.py --output results/HAT_DIV2K --model_path experiments/pretrained_models/HAT-L_SRx4_ImageNet-pretrain.pth \
--input datasets/DIV2K/DIV2K_train_LR_bicubic_X4_first120

CUDA_VISIBLE_DEVICES=0 python infer_tlc.py --output results/HAT-L_pretrain_TLC \
--model_path experiments/pretrained_models/HAT-L_SRx4_ImageNet-pretrain.pth
