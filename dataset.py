
import os
import numpy as np
import PIL
from torch.utils.data import Dataset
from torchvision import transforms

class CelebaDataset(Dataset):
    def __init__(self, path, size=128, lim=10000):
        self.sizes = [size, size]
        items, labels = [], []

        for data in list(os.listdir(path))[:int(lim)]:
            item = os.path.join(path, data)
            items.append(item)
            labels.append(data)
        self.items = items
        self.labels = labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = PIL.Image.open(item).convert('RGB')
        img = transforms.Resize(self.sizes)(img)
        img = np.array(img)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        return img, self.labels[idx]
