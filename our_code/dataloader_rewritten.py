import os
from PIL import Image
from torch.utils.data import Dataset

class MonoDataset(Dataset):
    """
    Creates a dataset with left/right image pairs from a text file containing
    filename paths.
    """
    def __init__(self, data_path, filenames_file, dataset, mode, transform=None):
        self.data_path = data_path
        self.dataset = dataset
        self.mode = mode
        self.transform = transform

        # Read from filenames_file for left/right image paths
        self.left_img_paths = []
        self.right_img_paths = []
        filenames = open(filenames_file, 'r')
        while True:
            line = filenames.readline()
            if not line:
                break
            else:
                files = line.split()
                left_file = files[0]
                right_file = files[1]
                self.left_img_paths.append(os.path.join(data_path,left_file))
                self.right_img_paths.append(os.path.join(data_path,right_file))
        filenames.close()

    def __len__(self):
        return len(self.left_img_paths)

    def __getitem__(self, i):
        if self.mode == 'train':
            left_img = Image.open(self.left_img_paths[i])
            right_img = Image.open(self.right_img_paths[i])
            sample = {'left_img': left_img, 'right_img': right_img}
        else:
            sample = Image.open(self.left_img_paths[i])

        if self.transform != None:
            sample = self.transform(sample)
        return sample
