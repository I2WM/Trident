import cv2
from PIL import Image
import numpy as np

img_path1 = r'\01sense-super-resolution\01datasets\xfdata2025\train\HR\train_108.jpg'
img_path2 = r'\01sense-super-resolution\01datasets\xfdata2025\train\HR\train_113.jpg'

# 读取图像

def flip(img_path):
    image = cv2.imread(img_path)
    # 水平翻转（flipCode=1 表示水平翻转）
    flipped_image = cv2.flip(image, 1)
    # 保存结果
    cv2.imwrite('train_108_flip.jpg', flipped_image)


def mixup_images(img_path1, img_path2, alpha=0.4):
    # 加载图像并转换为 numpy 数组（归一化到 [0, 1]）
    img1 = np.array(Image.open(img_path1).convert('RGB'), dtype=np.float32) / 255.0
    img2 = np.array(Image.open(img_path2).convert('RGB'), dtype=np.float32) / 255.0

    # Resize 保证两图一样大（如果不是）
    if img1.shape != img2.shape:
        img2 = np.array(Image.fromarray((img2 * 255).astype(np.uint8)).resize((img1.shape[1], img1.shape[0])), dtype=np.float32) / 255.0

    # 采样一个混合比例 lam（lambda）来自 beta 分布
    lam = 0.6
    # MixUp 图像
    mixed_img = lam * img1 + (1 - lam) * img2

    # 将混合后的图像转换回 PIL 并返回
    mixed_img = Image.fromarray((mixed_img * 255).astype(np.uint8))

    mixed_img.save('train_108_113_mix.jpg')



if __name__ == '__main__':
    flip(img_path1)
    mixup_images(img_path1, img_path2)