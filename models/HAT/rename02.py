import os
rename_dir = 'results/01HAT_SRx4_ImageNet-pretrain/visualization/DIV2K_valid'

img_list = os.listdir(rename_dir)

for name in img_list:
    full_path = os.path.join(rename_dir, name)
    # new_name = name.replace('x4','')
    new_name = name.split('_')[0]+'x4.png'
    print("new_name: {}".format(new_name))
    new_path = os.path.join(rename_dir, new_name)
    os.rename(full_path, new_path)
