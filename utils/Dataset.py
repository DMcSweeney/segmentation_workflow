"""
Custom dataset class for building seg. models
"""
import os
import numpy as np
from torch.utils.data import Dataset



class customDataset(Dataset):
    def __init__(self, image_path, transforms, read_masks=False, normalise=True, window=400, level=1074, worldmatch=False):
        super().__init__()
        # If worldmatch=True, apply WM correction (-1024 HU)
        self.images = self.load_data(os.path.join(image_path, 'slices/'), worldmatch) #~ Path to directory containing images
        self.transforms = transforms #~ Minimum ToTensor()
        self.ids = self.images['id']
        self.normalise = normalise
        self.WL_norm = self.WL_norm(self.images, window=window, level=level)
        self.read_masks = read_masks
        
        if self.read_masks:
            self.masks = self.load_data(os.path.join(image_path, 'masks/'))
    
    @staticmethod
    def load_data(path, worldmatch=False):
        #* Expects contents of directory to be .npy (ID.npy)
        data_dict = {'slices': [], 'id': []}
        for file in os.listdir(path):
            if file.endswith('.npy'): #!!
                name = file.split('.')[0]
                data_dict['id'].append(name)
                slice_ = np.load(path + file)
                slice_ = np.squeeze(slice_)
                if worldmatch:
                    slice_ += 1024
                data_dict['slices'].append(slice_)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict

    @staticmethod
    def norm_inputs(data):
        #*Normalise inputs between [0, 1]
        return (data['slices'] - data['slices'].min())/(data['slices'].max()-data['slices'].min())

    @staticmethod
    def WL_norm(data, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(data['slices'], minval, maxval)
        wld -= minval
        wld /= window
        return wld

    @staticmethod
    def convert_threeChannel(img):
        #~ SHAPE: H x W x C
        return np.repeat(img, 3, axis=-1)

    def __len__(self):
        return len(self.images['id'])
    
    def __getitem__(self, index: int):
        pid = self.ids[index]
        #* Apply W/L normalisation
        if self.normalise:
            img = self.WL_norm[index, ..., np.newaxis]
        else:
            img = self.images['slices'][index, ..., np.newaxis]

        #* Convert to three channels if needed
        out_img = self.convert_threeChannel(img)
        
        if self.read_masks:
            if len(self.masks['slices'].shape) != 4:
                mask = self.masks['slices'][index, ..., np.newaxis]

            else:
                mask = self.masks['slices'][index]
                mask = np.argmax(mask, axis=-1)

        if self.transforms:
            if self.read_masks:
                augmented = self.transforms(image=out_img, mask=mask)
                sample = {'inputs': augmented['image'], 
                        'targets': augmented['mask'],
                        'id': pid}
                return sample
            else:
                augmented = self.transforms(image=out_img)
                sample = {'inputs': augmented['image'],
                        'id': pid}
                return sample
        else:
            print('Need some transforms - minimum ToTensor().')
            raise

