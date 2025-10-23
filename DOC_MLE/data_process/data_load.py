import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import cv2
import albumentations as A
from DOC_MLE.data_process.retinal_process import rgb2gray, clahe_equalized, dataset_normalized, adjust_gamma
from DOC_MLE.data_process.data_ultils import group_images, visualize, label2rgb
from DOC_MLE import Constants
from DOC_MLE.data_process.data_ultils import  data_shuffle
import warnings
warnings.filterwarnings("ignore")

save_drive = '_drive'
save_drive_color = '_drive_color'
save_mo = '_mo'
save_pylop = '_pylop'
save_tbnc = '_tbnc'


def visual_sample(images, mask, path, per_row =5):
    visualize(group_images(images, per_row), Constants.visual_samples + path + '0')
    visualize(group_images(mask, per_row), Constants.visual_samples + path + '1')

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def get_drive_data(val_ratio = 1, is_train = True):
    images_piexllabel = load_from_npy(Constants.path_image_drive_piexllabel)
    mask_piexllabel = load_from_npy(Constants.path_label_drive_piexllabel)

    images_unlabel = load_from_npy(Constants.path_image_drive_unlabel)
    mask_unlabel = load_from_npy(Constants.path_label_drive_unlabel) #不用

    images_test = load_from_npy(Constants.path_test_image_drive)
    mask_test = load_from_npy(Constants.path_test_label_drive)

    images_piexllabel = rgb2gray(images_piexllabel)
    images_piexllabel = dataset_normalized(images_piexllabel)
    images_piexllabel = clahe_equalized(images_piexllabel)
    images_piexllabel = adjust_gamma(images_piexllabel, 1.0)


    images_unlabel = rgb2gray(images_unlabel)
    images_unlabel = dataset_normalized(images_unlabel)
    images_unlabel = clahe_equalized(images_unlabel)
    images_unlabel = adjust_gamma(images_unlabel, 1.0)

    images_piexllabel = images_piexllabel / 255.  # reduce to 0-1 rang
    images_unlabel = images_unlabel / 255.

    print(images_piexllabel.shape, mask_piexllabel.shape, '=================', np.max(images_piexllabel), np.max(mask_piexllabel))
    print('========  success load all Drive files ==========')
    visual_sample(images_piexllabel[0:10,:,:,:,], mask_piexllabel[0:10,:,:,:,], save_drive)
    val_num = int(images_test.shape[0] * val_ratio)
    # train_list = [images[val_num:, :, :, :, ], mask[val_num:, :, :, :, ]]
    # train_list = [images[0:, :, :, :, ], mask1[0:, :, :, :, ], mask2[0:, :, :, :, ]]
    train_list = [images_piexllabel[0:, :, :, :, ], mask_piexllabel[0:, :, :, :, ],images_unlabel[0:, :, :, :, ], mask_unlabel[0:, :, :, :, ]]

    val_list = [images_test[0:val_num, :, :, :, ], mask_test[0:val_num, :, :, :, ]]
    if is_train is True:
        return train_list, val_list
    else:
        return images_test, mask_test


class ImageFolder(data.Dataset):
    '''
    image is RGB original image, mask is one hot GT and label is grey image to visual
    img and mask is necessary while label is alternative
    '''

    def __init__(self, img1, mask1, img2, mask2, label=None):
        self.img1 = img1
        self.mask1 = mask1
        self.img2 = img2
        self.mask2= mask2
        self.label = label

    def __getitem__(self, index):
        imgs1 = torch.from_numpy(self.img1[index]).float()
        masks1 = torch.from_numpy(self.mask1[index]).float()
        imgs2 = torch.from_numpy(self.img2[index]).float()
        masks2 = torch.from_numpy(self.mask2[index]).float()


        if self.label is not None:
            label = torch.from_numpy(self.label[index]).float()

            return imgs1, masks1, imgs2, masks2, label
        else:
            return imgs1, masks1, imgs2, masks2

    def __len__(self):
        assert self.img1.shape[0] == self.mask1.shape[0], 'The number of images must be equal to labels'
        return self.img1.shape[0]



if __name__ == '__main__':

    get_drive_data()
    # get_monuclei_data()
    # get_MRI_chaos_data()
    # get_test_MRI_chaos_data()
    # get_tnbc_data(0.2, is_train = True)
    # get_pylyp_data()
    # get_drive_color_data()

    pass
