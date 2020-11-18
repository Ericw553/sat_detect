import logging
import os

import numpy as np
from numpy import moveaxis
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.DEBUG)


class _ImageMaskGenerator(Dataset):
    """Generator that provides batches of npy files from a directory."""

    def __init__(self, dirpath, batch_size, rescale=1.0, horizontal_flip=False,
                 data_format='channels_first', unpad_mask=0):
        """
        unpad_mask: how much to crop each mask by
        """
        self.image_dir = os.path.join(dirpath, 'images')
        self.mask_dir = os.path.join(dirpath, 'masks')
        self.batch_size = batch_size
        self.rescale = rescale
        self.horizontal_flip = horizontal_flip
        self.data_format = data_format
        self.unpad_mask = unpad_mask
        if data_format == 'channels_last':
            self.horiz_axis = -1
        else:
            self.horiz_axis = -3
        self.filenames = sorted(os.listdir(self.image_dir))
        self.filenames = [fn for fn in self.filenames
                          if os.path.splitext(fn)[-1].lower() == '.npy']

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Given index of batch, return batch of images."""
        imarrs = []
        maskarrs = []
        j = idx * self.batch_size
        for i in range(self.batch_size):
            fn = self.filenames[j % len(self.filenames)]
            img = np.load(os.path.join(self.image_dir, fn))
            mask = np.load(os.path.join(self.mask_dir, fn))
            if self.unpad_mask > 0:
                if self.data_format == 'channels_last':
                    mask = mask[...,
                           self.unpad_mask:-self.unpad_mask,
                           self.unpad_mask:-self.unpad_mask,
                           :]
                else:
                    img = moveaxis(img, 2, 0)
                    mask = moveaxis(mask, 2, 0)
            imarrs.append(img)
            maskarrs.append(mask)
        return np.array(imarrs[0]).astype(np.float32), np.array(maskarrs[0]).astype(np.float32)


class MyDataLoader:
    num_workers = 1

    def __init__(self, data_dir, batch_size: int, num_workers: int):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dir = data_dir + '/train'
        self.val_dir = data_dir + '/val'
        self.test_dir = data_dir + '/test'
        self.data_format = "channels_first"
        self.unpad = 92
        self.no_flip = True

    def dataloader(self):
        logging.info("Loading data")

        train_dataset = _ImageMaskGenerator(self.train_dir, batch_size=self.batch_size,
                                            data_format=self.data_format, unpad_mask=self.unpad,
                                            rescale=1. / 255, horizontal_flip=not self.no_flip)
        val_dataset = _ImageMaskGenerator(self.val_dir, batch_size=self.batch_size,
                                          data_format=self.data_format, unpad_mask=self.unpad,
                                          rescale=1. / 255)
        test_dataset = _ImageMaskGenerator(self.test_dir, batch_size=self.batch_size,
                                           data_format=self.data_format, unpad_mask=self.unpad,
                                           rescale=1. / 255)

        logging.debug("First batch image size (training): " + str(train_dataset[0][0].shape))
        logging.debug("First batch mask size (training): " + str(train_dataset[0][1].shape))

        logging.debug("First batch image size (valid): " + str(val_dataset[0][0].shape))
        logging.debug("First batch mask size (valid): " + str(val_dataset[0][1].shape))

        logging.debug("First batch image size (testing): " + str(test_dataset[0][0].shape))
        logging.debug("First batch mask size (testing): " + str(test_dataset[0][1].shape))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

        return train_dataloader, val_dataloader

    def dataloader_test(self):
        logging.info("Loading test data")

        test_dataset = _ImageMaskGenerator(self.test_dir, batch_size=self.batch_size,
                                           data_format=self.data_format, unpad_mask=self.unpad,
                                           rescale=1. / 255)

        logging.debug("First batch image size (testing): " + str(test_dataset[0][0].shape))
        logging.debug("First batch mask size (testing): " + str(test_dataset[0][1].shape))

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

        return test_dataloader
