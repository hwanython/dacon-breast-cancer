import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage import measure, morphology

def remove_small_objects(masks):

    masks = np.array(masks, dtype=np.uint8)
    all_labels = measure.label(masks, background=0)
    areas = [r.area for r in measure.regionprops(all_labels)]

    if len(areas)>=1:
        areas.sort()

        max_area1 = areas[0]

        out1 = morphology.remove_small_objects(all_labels, max_area1 - 1)

        # FIXME
        out1[out1 > 0] = 1
        out1 = out1.astype('uint8')
        del masks

        return out1

    return masks
def masked_process(data):
    ori_img2 = cv2.imread(data, 1)
    ori_img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)

    # otsu
    ret, th = cv2.threshold(ori_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # median filter
    dst = cv2.medianBlur(th, 5)
    out = remove_small_objects(dst)
    out = cv2.medianBlur(out, 5)
    mask = 1 - out
    masked_image = cv2.bitwise_and(ori_img2, ori_img2, mask=mask)
    return masked_image

if __name__ == '__main__':
    data_path = r'D:\jaehwan\98.dacon\mammography\data\open\train_imgs'
    target_path = r'D:\jaehwan\98.dacon\mammography\data\npy\train_imgs'
    data_list = os.listdir(data_path)
    im_size = 512
    uid_list = [uid.split('.')[0] for uid in data_list]

    tbar = tqdm(uid_list, ncols=130)
    for idx, uid in enumerate(tbar):
        img_path = os.path.join(data_path, uid+'.png')
        processed_img = masked_process(img_path)

        re_img = cv2.resize(processed_img, (im_size, im_size))
        # k = np.where(img > 200)
        # img[k] = 0
        np.save(os.path.join(target_path, uid+'.npy'), re_img)
