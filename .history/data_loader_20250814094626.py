import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class BreastUltrasoundDataset(Dataset):
    def __init__(self, path, images, masks, transform=None):
        self.path = 'Dataset_BUSI_with_GT/*/*'
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

def load_image(path, size):
    img = cv2.imread(path)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    return img

def load_data(root_path, size):
    # Count images & masks per class
    paths = glob(root_path)

    print(f"'normal' class has {len([i for i in paths if 'normal' in i and 'mask' not in i])} images and {len([i for i in paths if 'normal' in i and 'mask' in i])} masks.")
    print(f"'benign' class has {len([i for i in paths if 'benign' in i and 'mask' not in i])} images and {len([i for i in paths if 'benign' in i and 'mask' in i])} masks.")
    print(f"'malignant' class has {len([i for i in paths if 'malignant' in i and 'mask' not in i])} images and {len([i for i in paths if 'malignant' in i and 'mask' in i])} masks.")
    print(f"\nThere are total of {len([i for i in paths if 'mask' not in i])} images and {len([i for i in paths if 'mask' in i])} masks.")

    # Proceed with actual loading
    paths = sorted(glob(root_path))
    images = []
    masks = []
    x = 0
    for path in paths:
        img = load_image(path, size)
        if "mask" in path:
            if x:
                masks[-1] += img
                masks[-1] = (masks[-1] > 0.5).astype(float)
            else:
                masks.append(img)
                x = 1
        else:
            images.append(img)
            x = 0

    return np.array(images), np.array(masks)


def get_dataloaders(self, cfg):
    X, y = load_data(self.path, cfg.img_size)
    
    # Dropping normal class because it doesn't have a mask
    X = X[:647]
    y = y[:647]

    # Normalize & add channel dimension done in Dataset

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = BreastUltrasoundDataset(X_train, y_train)
    val_dataset = BreastUltrasoundDataset(X_val, y_val)
    test_dataset = BreastUltrasoundDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
