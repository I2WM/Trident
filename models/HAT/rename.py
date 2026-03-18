import os
rename_dir = r'\01sense-super-resolution\02code\HAT\results\001xf25_test_HAT-L_SRx4_finetune_from_ImageNet_pretrain_th_tta\visualization\XF2025_TEST'

img_list = os.listdir(rename_dir)

for name in img_list:
    full_path = os.path.join(rename_dir, name)
    new_name = name.replace('_low_hat','_sr').replace('png','jpg')
    new_path = os.path.join(rename_dir, new_name)
    os.rename(full_path, new_path)
