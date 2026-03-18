import os
import argparse
import re


def rename_images_remove_mamba(folder_path, pattern='_mamba', dest='.JPG'):
    """
    将文件夹中所有图片文件名中的指定模式替换为空字符

    参数:
        folder_path: 图片文件夹路径
        pattern: 要替换的模式，默认为 '_mamba'
    """
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在!")
        return

    # 获取文件夹中所有文件
    all_files = os.listdir(folder_path)

    # 过滤出图片文件
    image_files = [f for f in all_files
                   if os.path.isfile(os.path.join(folder_path, f)) and
                   any(f.lower().endswith(ext) for ext in image_extensions)]

    # image_files = [f for f in all_files]

    if not image_files:
        print(f"在文件夹 '{folder_path}' 中未找到图片文件!")
        return

    print(f"找到 {len(image_files)} 个图片文件")
    print(f"开始替换 '{pattern}' 为空字符...")

    renamed_count = 0

    for filename in image_files:
        # 检查文件名是否包含要替换的模式
        if pattern in filename:
            # 构建新文件名
            new_filename = filename.replace(pattern, dest)

            # 完整的旧文件路径和新文件路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # 检查新文件名是否已存在
            if os.path.exists(new_path):
                print(f"跳过 '{filename}': 新文件名 '{new_filename}' 已存在")
                continue

            try:
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"重命名 '{filename}' 时出错: {e}")
        else:
            print(f"跳过 '{filename}': 不包含 '{pattern}'")

    print(f"\n完成! 成功重命名了 {renamed_count} 个文件")
    print(f"跳过了 {len(image_files) - renamed_count} 个文件")


def main():
    parser = argparse.ArgumentParser(description='重命名图片文件，移除文件名中的指定模式')
    parser.add_argument('folder', nargs='*', help='包含图片的文件夹路径（可指定多个）')
    parser.add_argument('--pattern', default='_001_SRFormer_3DSR_from_pretrain_real_test.png',
                        help='要替换的模式，默认为 _mamba.png')
    parser.add_argument('--dest', default='.JPG',
                        help='要替换成的模式，默认为 .JPG')
    parser.add_argument('--dry-run', action='store_true',
                        help='模拟运行，不实际重命名文件')

    args = parser.parse_args()

    # 如果没有指定文件夹，使用默认的文件夹列表
    if not args.folder:
        default_folders = [
            r'D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_SRFormer_3DSR_from_pretrain_real_test\visualization\WestAccommodationAreas',
            r'D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_SRFormer_3DSR_from_pretrain_real_test\visualization\WestResearchAreas',
            r'D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_SRFormer_3DSR_from_pretrain_real_test\visualization\WestTeachingAreas',
            # r'D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_HAT-L_3DSRx4_from_pretrain_test_tta\visualization\WestAccommodationAreas',
            # r'D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_HAT-L_3DSRx4_from_pretrain_test_tta\visualization\WestResearchAreas',
            # r'D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_HAT-L_3DSRx4_from_pretrain_test_tta\visualization\WestTeachingAreas'
        ]
        folders = [f for f in default_folders if os.path.isdir(f)]

        if not folders:
            print("错误: 默认文件夹不存在!")
            return
    else:
        folders = args.folder

    for folder_path in folders:
        print(f"\n{'=' * 60}")
        print(f"处理文件夹: {folder_path}")
        print('=' * 60)

        if args.dry_run:
            print("模拟运行模式 (不会实际重命名文件):")

            # 检查文件夹
            if not os.path.isdir(folder_path):
                print(f"错误: 文件夹 '{folder_path}' 不存在!")
                continue

            # 支持的图片格式
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

            # 获取所有文件
            all_files = os.listdir(folder_path)
            image_files = [f for f in all_files
                           if os.path.isfile(os.path.join(folder_path, f)) and
                           any(f.lower().endswith(ext) for ext in image_extensions)]

            print(f"找到 {len(image_files)} 个图片文件")
            print(f"将替换 '{args.pattern}' 为空字符:")

            for filename in image_files:
                if args.pattern in filename:
                    # new_filename = filename.replace(args.pattern, '')
                    new_filename = filename.replace(args.pattern, args.dest)
                    print(f"  {filename} -> {new_filename}")
                else:
                    print(f"  {filename} -> (跳过: 不包含 '{args.pattern}')")
        else:
            rename_images_remove_mamba(folder_path, args.pattern, args.dest)


if __name__ == "__main__":
    main()

"""
python rename.py \
"D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_SRFormer_3DSR_from_pretrain_real_test\visualization\WestAccommodationAreas" \
"D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_SRFormer_3DSR_from_pretrain_real_test\visualization\WestResearchAreas" \
"D:\2026NTIRE\03SR\02results\3dsr\track1_lr\001_SRFormer_3DSR_from_pretrain_real_test\visualization\WestTeachingAreas" \
--pattern "_001_SRFormer_3DSR_from_pretrain_real_test.png" \
--dest ".JPG"
"""