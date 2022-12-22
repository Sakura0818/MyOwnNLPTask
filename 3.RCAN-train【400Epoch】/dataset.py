import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import numpy as np
import PIL.Image as pil_image
import torch




class Dataset(object):
    def __init__(self, images_dir, patch_size, scale, use_fast_loader=False):
        self.image_fileslr = sorted(glob.glob(images_dir +'/LR'+ '/*.npy'))
        self.image_fileshr = sorted(glob.glob(images_dir +'/HR'+ '/*.npy'))

        self.patch_size = patch_size
        self.scale = scale
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        
        #print("idx:{}".format(idx))
        idx = self._get_index(idx)
        #print("now is :{}".format(idx))
        lr = np.load(self.image_fileslr[idx])
        hr = np.load(self.image_fileshr[idx])
        lr,hr=get_patch(lr,hr,self.patch_size*self.scale,self.scale)#
        lr,hr=augment([lr,hr])
        lr_tensor, hr_tensor = np2Tensor([lr, hr], 255)
        #print("lr:{}".format(lr_tensor.size()))
        #print("hr:{}".format(hr_tensor.size()))


        # normalization
        lr_tensor /= 255.0
        hr_tensor /= 255.0

        return lr_tensor, hr_tensor

    def __len__(self):
        #print(len(self.image_fileshr))
        return len(self.image_fileshr)
    def _get_index(self, idx):
        
        return idx % len(self.image_fileshr)
        



def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        #可以这样认为，ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        #转换为  c height weight
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):#随机裁剪一块
    ih, iw = img_in.shape[:2]

    p = scale if multi_scale else 1
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]

