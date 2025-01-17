import os
import os.path

import cv2
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


IMAGE_DIR = "data/nyud/data/images"
SEG_DIR = "data/nyud/segmentation"

DATA_LIST = {
    "train": "data/nyud/data_list/train.txt",
    "trainval": "data/nyud/data_list/trainval.txt",
    "test": "data/nyud/data_list/test.txt",
    "val": "data/nyud/data_list/val.txt",
}


def make_dataset(split='train'):
    """

    :param split:
    :param data_root:
    :param data_list: the datalist dir
    :return:
    """
    assert split in ['train', 'val', 'test', 'trainval']
    data_list = DATA_LIST[split]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    with open(data_list) as f:
        list_read = f.readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = int(line.strip('\n'))
        image_name = os.path.join(IMAGE_DIR, f'img_{line}.png')
        label_name = os.path.join(SEG_DIR, f'img_{line}.mat')
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list



class SemData(Dataset):
    def __init__(self, split='train', transform=None):
        self.split = split
        self.data_list = make_dataset(split)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = loadmat(label_path)['segmentation']  # GRAY 1 channel ndarray with shape H * W

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label
