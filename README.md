# [3D-Super-Resolution](https://github.com/I2WM/3D-Super-Resolution)



### Track1

#### Instruction

We use SRFormerV2 to perform super-resolution of rendered LR in Track1

#### Execution

```bash
# activate SRFormerV2
conda activate srformer

# SRFormerV2 inference
python models/SRFormer/basicsr/test.py -opt models/SRFormer/options/test/SRFormerV2/002_SRFormer_3DSR_from_pretrain_test_final.yml

```


### Track2

#### Instruction
HAT was pretrained on DIV2K and Flickr2K, we also finetuned HAT models on OST and LSDIR dataset.

#### Execution

```bash
# activate mambairv2
conda activate mambair

# MambaIRv2 inference
python models/MambaIR/basicsr/test.py -opt models/MambaIR/options/test/mambairv2/004_My_3DSR_MambaIRv2_RealSR_x4_tta_test.yml

# Rename the inference results
python /models/MambaIR/rename.py \
"/path/to/track2/EastResearchAreas" \
"/path/to/track2/NorthAreas" \
--pattern "_mamba.png" \
--dest ".JPG"

# SRFormer
conda activate srformer

# SRFormer inference
python models/SRFormer/basicsr/test.py -opt models/SRFormer/options/test/SRFormerV2/002_SRFormer_3DSR_from_pretrain_real_test_final.yml

# Rename the inference results
python /models/MambaIR/rename.py \
"/path/to/track2/EastResearchAreas" \
"/path/to/track2/NorthAreas" \
--pattern "_002_SRFormer_3DSR_from_pretrain_real_test_final.png" \
--dest ".JPG"

# HAT-L
conda activate HAT

# HAT-L inference
torchrun --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29511 \
  /models/HAT/hat/test.py -opt /models/HAT/options/test/NTIRE2026/002_HAT-L_3DSRx4_from_pretrain_real_test_final.yml \
  --launcher pytorch

# Rename the inference results
python /models/MambaIR/rename.py \
"/path/to/track2/EastResearchAreas" \
"/path/to/track2/NorthAreas" \
--pattern "_hat.png" \
--dest ".JPG"
  
# Model Ensemble
# 0.06 MambaIR+0.02SRFormer+0.92HAT
conda activate HAT

# 1， Conduct model ensemble for EastResearchAreas
python /models/HAT/ensemble.py \
--folder1 /path/to/HAT/track2/EastResearchAreas \
--folder2 /path/to/HAT/track2/EastResearchAreas \
--folder3 /path/to/HAT/track2/EastResearchAreas \
--weights 0.06 0.02 0.92 \
--output /path/to/track2/EastResearchAreas/rgb

# Conduct model ensemble for NorthAreas
python /models/HAT/ensemble.py \
--folder1 /path/to/MambaIR/track2/NorthAreas \
--folder2 /path/to/SRFormer/track2/NorthAreas \
--folder3 /path/to/HAT/track2/NorthAreas \
--weights 0.06 0.02 0.92 \
--output /path/to/track2/NorthAreas/rgb

```