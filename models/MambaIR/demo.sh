# test
python basicsr/test.py -opt options/test/mambairv2/001_My_test_MambaIRv2_SRLarge_x4.yml

# test
python basicsr/test.py -opt options/test/mambairv2/001_My_test_MambaIRv2_SRLarge_x4_tta.yml
python basicsr/test.py -opt options/test/mambairv2/003_My_test_DIV2K_MambaIRv2_SRLarge_x4_tta.yml
python basicsr/test.py -opt options/test/mambairv2/004_My_test_DIV2K_MambaIRv2_SRLarge_x4_tta.yml
CUDA_VISIBLE_DEVICES=1 python basicsr/test.py -opt options/test/mambairv2/005_My_test_DIV2K_MambaIRv2_SRLarge_x4_tta.yml
CUDA_VISIBLE_DEVICES=1 python basicsr/test.py -opt options/test/mambairv2/002_My_test_denoise_MambaIRv2_tta_hx.yml

CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/mambairv2/001_My_3DSR_MambaIRv2_SRLarge_x4_tta.yml


# test denoise
CUDA_VISIBLE_DEVICES=1 python basicsr/test.py -opt options/test/mambairv2/002_My_test_denoise_MambaIRv2_tta_hx.yml
python basicsr/test.py -opt options/test/mambairv2/003_My_test_denoise_MambaIRv2_tta_hx.yml
python basicsr/test.py -opt options/test/mambairv2/004_My_test_denoise_MambaIRv2_tta_hx.yml


# train on denosing50, 3090, 28min/val
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2414 basicsr/train.py \
-opt options/train/mambairv2/001_My_train_MambaIRv2_DIV2K_level50.yml --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=1 --master_port=2414 basicsr/train.py \
-opt options/train/mambairv2/002_My_train_MambaIRv2_DIV2K_level50_hx.yml --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=1 --master_port=2414 basicsr/train.py \
-opt options/train/mambairv2/002_My_train_MambaIRv2_DIV2K_level50_hx.yml --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=1 --master_port=2414 basicsr/train.py \
-opt options/train/mambairv2/003_My_train_MambaIRv2_DIV2K_level50_hx.yml --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=1 --master_port=2414 basicsr/train.py \
-opt options/train/mambairv2/004_My_train_MambaIRv2_DIV2K_level50_hx.yml --launcher pytorch