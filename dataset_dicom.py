import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from coursework.coursework.utils import read_dicom_image, rle_decode


class PneumothoraxDicomDataset(Dataset):
    def __init__(self, dicom_map, rle_map, id_list, image_size=128):
        self.dicom_map = dicom_map
        self.rle_map = rle_map
        self.id_list = id_list
        self.image_size = image_size

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        image_id = self.id_list[index]

        dicom_path = self.dicom_map.get(image_id)
        if dicom_path is None:
            raise FileNotFoundError(f"dcm file for image id not found: {image_id}")

        image = read_dicom_image(dicom_path)

        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        rles = self.rle_map.get(image_id, [])
        for rle in rles:
            decoded = rle_decode(rle, (h, w))
            mask = np.maximum(mask, decoded)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)
        # mask = (mask > 0.5).astype(np.float32) # todo
        mask = (mask > 0.2).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return image_tensor, mask_tensor, image_id


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1.0 - dice


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return bce_loss + dice_loss