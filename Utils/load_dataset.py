import cv2
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ClassDataset(Dataset):
    def __init__(self, root, transform=None, suffix="dm", folder_h5= "my_dataset"):
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.suffix = suffix
        self.folder_h5 = folder_h5

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img, target = load_data(img_path, self.suffix, self.folder_h5)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target



def load_data(img_path, suffix, folder_h5):
    gt_path = img_path.replace('.jpg', suffix + '.h5').replace('images', folder_h5)
    img = Image.open(img_path).convert("RGB")
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file["density"])

    #this delete the logo in the pic
    img = np.array(img)
    cv2.circle(img, (1204, 633), 32, (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (1100, 672), (1265, 685), (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (1154, 685), (1265, 710), (0, 0, 0), thickness=-1)
    img = Image.fromarray(img)

    return img, target