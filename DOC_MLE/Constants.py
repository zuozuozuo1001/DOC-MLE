

#========================== public configure ==========================
# IMG_SIZE = (300, 300)
IMG_SIZE = (288, 288)
TOTAL_EPOCH = 600
INITAL_EPOCH_LOSS = 100000
NUM_EARLY_STOP = 60
NUM_UPDATE_LR = 100
BINARY_CLASS = 1
BATCH_SIZE = 2
learning_rates= 1e-3

# ===================   DRIVE configure =========================
DATA_SET = 'DCA1'
visual_samples = '/root/daima/zixie_weakly_semi/log/visual_samples/'
saved_path = '/root/daima/zixie_weakly_semi/log/weight_save/'+ DATA_SET + '/'
visual_results = '/root/daima/zixie_weakly_semi/log/visual_results/'+ DATA_SET + '/'


resize_drive = 288
resize_size_drive = (resize_drive, resize_drive)
size_h, size_w = 300, 300

# resize_drive = 288
# resize_size_drive = (resize_drive, resize_drive)
# size_h, size_w = 288, 288

path_image_drive_piexllabel='/root/daima/zixie_weakly_semi/dataset1/npy/DCA1/tempt_5_5/train_image_piexllabel_save.npy'
path_label_drive_piexllabel='/root/daima/zixie_weakly_semi/dataset1/npy/DCA1/tempt_5_5/train_label_piexllabel_save.npy'
path_image_drive_unlabel='/root/daima/zixie_weakly_semi/dataset1/npy/DCA1/tempt_5_5/train_image_unlabel_save.npy'
path_label_drive_unlabel='/root/daima/zixie_weakly_semi/dataset1/npy/DCA1/tempt_5_5/train_label_unlabel_save.npy'
path_test_image_drive='/root/daima/zixie_weakly_semi/dataset1/npy/DCA1/tempt_5_5/test_image_save.npy'
path_test_label_drive='/root/daima/zixie_weakly_semi/dataset1/npy/DCA1/tempt_5_5/test_label_save.npy'


total_drive = 34
Classes_drive_color = 5
###########################################################################################