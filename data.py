import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import torch


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask = mask / 255
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        else:
            return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv.resize(image, dsize=(self.W, self.H), interpolation=cv.INTER_LINEAR)
        mask = cv.resize(mask, dsize=(self.W, self.H), interpolation=cv.INTER_LINEAR)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, mode='train'):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)  # [:len(self.images)//10]
        self.gts = sorted(self.gts)  # [:len(self.gts)//10]
        self.filter_files()

        self.normalize = Normalize(mean=np.array([[[124.55, 118.90, 102.94]]]), std=np.array([[[56.77, 55.97, 57.50]]]))
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(trainsize, trainsize)
        self.totensor = ToTensor()

        self.mode = mode

    def __getitem__(self, index):
        image = cv.imread(self.images[index])[:, :, ::-1].astype(np.float32)
        mask = cv.imread(self.gts[index], 0).astype(np.float32)
        shape = mask.shape
        if self.mode == 'test':
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image.float(), mask.float(), self.gts[index].split('/')[-1], shape
        elif self.mode == 'val':
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image.float(), mask.float()
        else:
            image, mask = self.normalize(image, mask)
            # image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            return image, mask

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def collate(self, batch):
        size = 352  # [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv.resize(image[i], dsize=(size, size), interpolation=cv.INTER_LINEAR)
            mask[i] = cv.resize(mask[i], dsize=(size, size), interpolation=cv.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2).float()
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1).float()
        return image, mask

    def __len__(self):
        return len(self.images)


def get_loader(image_root, gt_root, batchsize, trainsize, mode='train', shuffle=True, num_workers=8, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, trainsize, mode=mode)
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory, collate_fn=dataset.collate)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
    return data_loader


class picture_process(object):

    def __init__(self, testsize):
        self.testsize = testsize
        self.normalize = Normalize(mean=np.array([[[124.55, 118.90, 102.94]]]), std=np.array([[[56.77, 55.97, 57.50]]]))
        self.resize = Resize(testsize, testsize)
        self.totensor = ToTensor()

    def get_data(self, image):
        image, mask = self.normalize(image, image[:, :, 0])
        image, mask = self.resize(image, mask)
        image, mask = self.totensor(image, mask)
        return image.float()
