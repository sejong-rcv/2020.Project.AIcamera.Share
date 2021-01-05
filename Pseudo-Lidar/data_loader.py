import os
from PIL import Image

from torch.utils.data import Dataset


class KAISTLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None,RGB=False):
        if mode=="val":
            mode_="test"
        else:
            mode_=mode
        txtpath=os.path.join(root_dir,"txt","%s.txt"%mode_)
        txt=open(txtpath,"r")
        self.left_paths=[]
        self.right_paths=[]
        self.thermal_paths=[]
        for line in txt:
            sp=line.split()
            self.left_paths.append(os.path.join(root_dir,sp[0]))
            self.right_paths.append(os.path.join(root_dir,sp[1]))
            self.thermal_paths.append(os.path.join(root_dir,sp[2]))
            
        self.transform = transform
        self.mode = mode
        self.RGB=RGB

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        thermal_image = Image.open(self.thermal_paths[idx]).convert("RGB")
        
        if self.mode == 'train'or self.mode == 'val':
            right_image = Image.open(self.right_paths[idx])
           
            sample = {'left_image': left_image, 'right_image': right_image, 'thermal_image': thermal_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                if self.RGB:
                    left_image = self.transform(left_image)
                else:
                    left_image = self.transform(thermal_image)
                
            return left_image
