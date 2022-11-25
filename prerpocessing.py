import os
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    data_path = r'D:\jaehwan\98.dacon\mammography\data\open\train_imgs'
    target_path = r'D:\jaehwan\98.dacon\mammography\data\npy\train_imgs'
    data_list = os.listdir(data_path)
    im_size = 512
    uid_list = [uid.split('.')[0] for uid in data_list]

    tbar = tqdm(uid_list, ncols=130)
    for idx, uid in enumerate(tbar):
        img = cv2.imread(os.path.join(data_path, uid+'.png'))
        img = cv2.resize(img, (im_size, im_size))
        # k = np.where(img > 200)
        # img[k] = 0
        np.save(os.path.join(target_path, uid+'.npy'), img)
