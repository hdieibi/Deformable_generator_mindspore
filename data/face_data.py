from mindspore.dataset import transforms, vision
import os
from PIL import Image


class FaceData:
    def __init__(self, dataset_path, mode='train'):
        super().__init__()
        assert mode in ['train', 'eval']
        self.mode_path = os.path.join(dataset_path, mode)
        self.data_list = os.listdir(self.mode_path)
        self.transform = transforms.Compose([
            vision.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.mode_path, self.data_list[index])
        img = Image.open(img_path)
        image = self.transform(img)
        return image

    def __len__(self):
        return len(self.data_list)