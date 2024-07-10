from torch.utils.data import Dataset
import torch
import os
from torchvision.utils import save_image
from PIL import Image
import torchvision

totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()

class Data_set(Dataset):
    def __init__(self, opt, phase=None):
        super(Data_set, self).__init__()
        self.opt = opt
        self.imgs_paths = self.opt['dataset_path']
        self.dataset_paths = self.get_dataset_from_txt_file(self.opt['dataroot'])
        self.dataset_size = len(self.dataset_paths)
        self.gt_dir_root = self.opt['gt_dataset_path']
        self.cond_dir_root = self.opt['cond_dataset_path']
        self.split = phase

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, item):
        return self.pull_item(item)
    
    @property
    def num_samples(self):
        return self.dataset_size

    def load_file(self, gt_file_path, cond_file_path):
        gt = Image.open(gt_file_path).convert("RGB")
        cond = Image.open(cond_file_path).convert("RGB")
        return gt, cond
    
    def pull_item(self, item):
        gt_file_path = os.path.join(self.gt_dir_root, self.dataset_paths[item])
        cond_file_path = os.path.join(self.cond_dir_root, self.dataset_paths[item])
        gt_img, cond_img = self.load_file(gt_file_path, cond_file_path)
        [gt_img, cond_img] = self.transform_augment([gt_img, cond_img], split=self.split, min_max=(-1, 1))
        return {'GT': gt_img, 'condition': cond_img}
    
    def get_dataset_from_txt_file(self, file_path):
        with open(file_path, 'r') as f:
            content = f.readlines()
            return [i.strip() for i in content]
        
    def transform_augment(self, img_list, split='val', min_max=(0, 1)):    
        imgs = [totensor(img) for img in img_list]
        if split == 'train':
            imgs = torch.stack(imgs, 0)
            imgs = hflip(imgs)
            imgs = torch.unbind(imgs, dim=0)
        ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
        return ret_img