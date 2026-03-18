import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings('ignore')


def weighted_image_fusion(folder1, folder2, folder3, output_folder,
                          weights=(0.3, 0.3, 0.4),
                          extensions=None,
                          save_dtype=np.uint16,
                          verbose=True):
    """
    加权融合三个文件夹中的同名图片

    Args:
        folder1: 第一个文件夹路径
        folder2: 第二个文件夹路径
        folder3: 第三个文件夹路径
        output_folder: 输出文件夹路径
        weights: 三个文件夹的权重，默认为 (0.3, 0.3, 0.4)
        extensions: 支持的图片扩展名列表
        save_dtype: 保存的数据类型 (np.uint8 或 np.uint16)
        verbose: 是否显示详细信息
    """
    # 确保权重和为1
    weights = np.array(weights)
    weights = weights / weights.sum()

    if verbose:
        print(f"权重归一化后: {weights}")

    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 默认支持的图片格式
    if extensions is None:
        extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp', '.tiff', '.tif'}

    # 获取三个文件夹中的所有文件
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    folder3_path = Path(folder3)

    # 获取第一个文件夹中的所有图片（作为基准）
    image_files = []
    for ext in extensions:
        image_files.extend(folder1_path.glob(f"*{ext}"))
        image_files.extend(folder1_path.glob(f"*{ext.upper()}"))

    # 去重并排序
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"在 {folder1} 中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片需要处理")

    # 统计信息
    stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'total': len(image_files)
    }

    # 处理每张图片
    for img_path in tqdm(image_files, desc="融合处理"):
        filename = img_path.name

        # 检查其他两个文件夹中是否存在同名文件
        img2_path = folder2_path / filename
        img3_path = folder3_path / filename

        if not img2_path.exists() or not img3_path.exists():
            if verbose:
                missing = []
                if not img2_path.exists():
                    missing.append(str(img2_path))
                if not img3_path.exists():
                    missing.append(str(img3_path))
                print(f"跳过 {filename}: 缺失文件 {missing}")
            stats['skipped'] += 1
            continue

        try:
            # 读取三张图片
            img1 = Image.open(img_path)
            img2 = Image.open(img2_path)
            img3 = Image.open(img3_path)

            # 检查图片尺寸是否一致
            if img1.size != img2.size or img1.size != img3.size:
                if verbose:
                    print(f"警告: {filename} 尺寸不一致，将调整到相同尺寸")
                # 统一调整到最小尺寸
                min_width = min(img1.width, img2.width, img3.width)
                min_height = min(img1.height, img2.height, img3.height)

                if img1.size != (min_width, min_height):
                    img1 = img1.resize((min_width, min_height), Image.BICUBIC)
                if img2.size != (min_width, min_height):
                    img2 = img2.resize((min_width, min_height), Image.BICUBIC)
                if img3.size != (min_width, min_height):
                    img3 = img3.resize((min_width, min_height), Image.BICUBIC)

            # 检查图片模式是否一致
            modes = {img1.mode, img2.mode, img3.mode}
            if len(modes) > 1:
                if verbose:
                    print(f"警告: {filename} 模式不一致 {modes}，将统一转换为 RGB")
                # 统一转换为 RGB
                target_mode = 'RGB'
                if img1.mode != target_mode:
                    img1 = img1.convert(target_mode)
                if img2.mode != target_mode:
                    img2 = img2.convert(target_mode)
                if img3.mode != target_mode:
                    img3 = img3.convert(target_mode)

            # 转换为 numpy 数组，保持高精度
            arr1 = np.array(img1).astype(np.float32)
            arr2 = np.array(img2).astype(np.float32)
            arr3 = np.array(img3).astype(np.float32)

            # 检查像素值范围
            max_val = max(arr1.max(), arr2.max(), arr3.max())
            if max_val > 255 and save_dtype == np.uint8:
                if verbose:
                    print(f"警告: {filename} 像素值超过255，将归一化后保存为 uint8")
                # 归一化到 [0, 255]
                arr1 = (arr1 / max_val * 255).clip(0, 255)
                arr2 = (arr2 / max_val * 255).clip(0, 255)
                arr3 = (arr3 / max_val * 255).clip(0, 255)

            # 加权融合
            fused = weights[0] * arr1 + weights[1] * arr2 + weights[2] * arr3

            # 根据保存类型处理
            if save_dtype == np.uint16:
                # 如果原始数据是16位，保持16位
                if max_val > 255:
                    fused = np.clip(fused, 0, max_val).astype(np.uint16)
                else:
                    fused = np.clip(fused, 0, 255).astype(np.uint16) * 257  # 扩展到16位范围
            else:
                # 保存为8位
                fused = np.clip(fused, 0, 255).astype(np.uint8)

            # 保存融合后的图片（无损）
            output_file = output_path / filename

            # 根据原始文件扩展名选择保存格式
            ext = img_path.suffix.lower()

            # 无损保存参数
            save_params = {}
            if ext == '.png':
                save_params = {'compress_level': 0}  # PNG无压缩
            elif ext in ['.jpg', '.jpeg']:
                # 如果保存为JPEG，会损失质量，建议保存为PNG保持无损
                # output_file = output_path / f"{img_path.stem}.png"
                output_file = output_path / f"{img_path.stem}.JPG"
                if verbose:
                    print(f"  将 {filename} 保存为 PNG 以保持无损")
            elif ext == '.tiff':
                save_params = {'compression': None}  # TIFF无压缩

            # 创建图像并保存
            if fused.ndim == 2:  # 灰度图
                result_img = Image.fromarray(fused, mode='L' if save_dtype == np.uint8 else 'I;16')
            else:  # 彩色图
                if save_dtype == np.uint16:
                    # PIL 不支持直接保存 16位彩色 PNG，需要特殊处理
                    import imageio
                    imageio.imwrite(output_file, fused)
                else:
                    result_img = Image.fromarray(fused)
                    result_img.save(output_file, **save_params)

            # 如果使用了 imageio，跳过重复保存
            if not (save_dtype == np.uint16 and fused.ndim == 3):
                result_img.save(output_file, **save_params)

            stats['processed'] += 1

            # 可选：显示示例结果
            if verbose and stats['processed'] <= 5:
                print(f"\n处理: {filename}")
                print(f"  原始尺寸: {img1.size}")
                print(f"  像素范围: [{fused.min():.0f}, {fused.max():.0f}]")
                print(f"  均值: {fused.mean():.2f}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            stats['failed'] += 1

    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总图片数: {stats['total']}")
    print(f"成功融合: {stats['processed']}")
    print(f"跳过: {stats['skipped']}")
    print(f"失败: {stats['failed']}")
    print(f"\n输出文件夹: {output_path}")

    # 输出权重信息
    print(f"\n融合权重: Model1={weights[0]:.3f}, Model2={weights[1]:.3f}, Model3={weights[2]:.3f}")

    return stats


def batch_verify_fusion(output_folder, sample_count=5):
    """
    验证融合结果
    """
    output_path = Path(output_folder)
    images = list(output_path.glob("*.png"))[:sample_count]

    if not images:
        print("没有找到融合后的图片")
        return

    print("\n" + "=" * 60)
    print("融合结果验证")
    print("=" * 60)

    for img_path in images:
        img = Image.open(img_path)
        arr = np.array(img)

        print(f"\n{img_path.name}:")
        print(f"  尺寸: {img.size}")
        print(f"  模式: {img.mode}")
        print(f"  数据类型: {arr.dtype}")
        print(f"  像素范围: [{arr.min()}, {arr.max()}]")
        print(f"  均值: {arr.mean():.2f}")
        print(f"  标准差: {arr.std():.2f}")


def main():
    parser = argparse.ArgumentParser(description='加权融合三个文件夹中的同名图片')
    parser.add_argument('--folder1', '-1', type=str, required=True,
                        help='第一个文件夹路径')
    parser.add_argument('--folder2', '-2', type=str, required=True,
                        help='第二个文件夹路径')
    parser.add_argument('--folder3', '-3', type=str, required=True,
                        help='第三个文件夹路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出文件夹路径')
    parser.add_argument('--weights', '-w', type=float, nargs=3, default=[0.3, 0.3, 0.4],
                        help='三个权重值，默认为 0.3 0.3 0.4')
    parser.add_argument('--uint16', action='store_true', default=False,
                        help='保存为16位深度（无损）')
    parser.add_argument('--verify', action='store_true',
                        help='完成后验证结果')

    args = parser.parse_args()

    print("=" * 60)
    print("模型集成 - 加权图片融合")
    print("=" * 60)
    print(f"文件夹1: {args.folder1}")
    print(f"文件夹2: {args.folder2}")
    print(f"文件夹3: {args.folder3}")
    print(f"输出文件夹: {args.output}")
    print(f"权重: {args.weights}")
    print(f"保存格式: {'16位' if args.uint16 else '8位'}")
    print("=" * 60)

    # 执行融合
    save_dtype = np.uint16 if args.uint16 else np.uint8
    stats = weighted_image_fusion(
        folder1=args.folder1,
        folder2=args.folder2,
        folder3=args.folder3,
        output_folder=args.output,
        weights=args.weights,
        save_dtype=save_dtype
    )

    # 验证结果
    if args.verify and stats['processed'] > 0:
        batch_verify_fusion(args.output)


# 简单使用版本
def simple_fusion():
    """
    简单使用示例
    """
    # 配置路径
    folder1 = "./model_outputs/model1"
    folder2 = "./model_outputs/model2"
    folder3 = "./model_outputs/model3"
    output_folder = "./ensemble_results"

    # 设置权重（可以根据模型性能调整）
    weights = (0.25, 0.25, 0.5)  # 第三个模型权重更高

    print(f"开始融合...")
    print(f"权重: {weights}")

    stats = weighted_image_fusion(
        folder1=folder1,
        folder2=folder2,
        folder3=folder3,
        output_folder=output_folder,
        weights=weights,
        # save_dtype=np.uint16,  # 使用16位保存，完全无损
        save_dtype=np.uint8,  # 使用8位保存
        verbose=True
    )

    # 验证结果
    if stats['processed'] > 0:
        batch_verify_fusion(output_folder)


if __name__ == "__main__":
    # 使用命令行参数版本
    main()

    # 或者使用简单版本（修改路径后直接运行）
    # simple_fusion()