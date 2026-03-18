import cv2
from scipy.io import loadmat
import os
import numpy as np
from glob import glob

'''
=======================================train=======================================
avg_h: 1349.5951045385007 avg_w: 1360.0892401835797
=======================================test=======================================
avg_h: 859.85 avg_w: 857.8
'''



# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    # im decode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


if __name__ == '__main__':
    # 得读原图大小，要分块
    img_dir = r"\02free-range-chicken-sr\01datasets\test"
    # gt_dir = 'D:\PycharmProject\CrowdCounting-P2PNet\crowd_datasets\Dark_RGBT_CC'
    # pred_dir = 'pred'

    # avg_resolution = {}
    resolution_list = []
    h_list = []
    w_list = []
    num_list = []


    sub_img_dir = os.path.join(img_dir)

    # analyse images
    im_list = os.listdir(sub_img_dir)
    # print("im_list: {}".format(im_list))
    for im_path in im_list:
        if 'README' in im_path:
            continue
        all_path = os.path.join(sub_img_dir, im_path)
        print("all_path: {}".format(all_path))
        # im = cv2.imread(all_path)
        im = cv_imread(all_path)
        (h,w,c) = im.shape
        # print("h,w: {},{}".format(h,w))
        resolution_list.append((h,w))
        h_list.append(h)
        w_list.append(w)


    print("h_list: {}".format(h_list))
    print("w_list: {}".format(w_list))


    h_list = np.array(h_list)
    w_list = np.array(w_list)

    avg_h = np.mean(h_list)
    avg_w = np.mean(w_list)
    # total_num = np.sum(num_list)
    # avg_num = np.mean(num_list)
    # min_num = np.min(num_list)
    # max_num = np.max(num_list)



    # print the evaluation results
    print('=======================================test=======================================')
    print("avg_h:", avg_h, "avg_w:", avg_w)
    # print("avg_num:", avg_num, "min_num:", min_num, "max_num:", max_num, "total_num:", total_num)
    print('=======================================test=======================================')

