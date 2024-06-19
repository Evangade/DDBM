import os.path
import torch
import random
import numpy as np
from .image_folder import make_dataset
from PIL import Image

import torchvision
import blobfile as bf

from glob import glob
from torchvision import transforms

def get_params( size,  resize_size,  crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}
 

def get_transform(params,  resize_size,  crop_size, method=Image.BICUBIC,  flip=True, crop = True, totensor=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale(img, crop_size, method)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __scale(img, target_width, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        return torch.nn.functional.interpolate(img.unsqueeze(0), size=(target_width, target_width), mode='bicubic', align_corners=False).squeeze(0)
    else:
        return img.resize((target_width, target_width), method)

def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    return __flip(img, flip)


class EdgesDataset(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True,  img_size=256, random_crop=False, random_flip=True):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        if train:
            self.train_dir = os.path.join(dataroot, 'train')  # get the image directory
            self.train_paths = make_dataset(self.train_dir) # get image paths
            self.AB_paths = sorted(self.train_paths)
        else:

            self.test_dir = os.path.join(dataroot, 'val')  # get the image directory
            
            self.AB_paths = make_dataset(self.test_dir) # get image paths
            
        self.crop_size = img_size
        self.resize_size = img_size
        
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        params =  get_params(A.size, self.resize_size, self.crop_size)

        transform_image = get_transform( params, self.resize_size, self.crop_size, crop =self.random_crop, flip=self.random_flip)

        A = transform_image(A)
        B = transform_image(B)

        if not self.train:
            return  B, A, index, AB_path
        else:
            return B, A, index

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)




class DIODE(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True,  img_size=256, random_crop=False, random_flip=True, down_sample_img_size = 0, cache_name='cache', disable_cache=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.image_root = os.path.join(dataroot, 'train' if train else 'val')
        self.crop_size = img_size
        self.resize_size = img_size
        
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train

        self.filenames = [l for l in os.listdir(self.image_root) if not l.endswith('.pth') and not l.endswith('_depth.png') and not l.endswith('_normal.png')]

        self.cache_path = os.path.join(self.image_root, cache_name+f'_{img_size}.pth')
        if os.path.exists(self.cache_path) and not disable_cache:
            self.cache = torch.load(self.cache_path)
            # self.cache['img'] = self.cache['img'][:256]
            self.scale_factor = self.cache['scale_factor']
            print('Loaded cache from {}'.format(self.cache_path))
        else:
            self.cache = None

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        
        fn = self.filenames[index]
        img_path = os.path.join(self.image_root, fn)
        label_path = os.path.join(self.image_root, fn[:-4]+'_normal.png')

        with bf.BlobFile(img_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        with bf.BlobFile(label_path, "rb") as f:
            pil_label = Image.open(f)
            pil_label.load()
        pil_label = pil_label.convert("RGB")

        # apply the same transform to both A and B
        params =  get_params(pil_image.size, self.resize_size, self.crop_size)

        transform_label = get_transform(params, self.resize_size, self.crop_size, method=Image.NEAREST, crop =False, flip=self.random_flip)
        transform_image = get_transform( params, self.resize_size, self.crop_size, crop =False, flip=self.random_flip)

        cond = transform_label(pil_label)
        img = transform_image(pil_image)

        # if self.down_sample_img:
        #     image_pil = np.array(image_pil).astype(np.uint8)
        #     down_sampled_image = self.down_sample_img(image=image_pil)["image"]
        #     down_sampled_image = get_tensor()(down_sampled_image)
        #     # down_sampled_image = transforms.ColorJitter(brightness = [0.85,1.15], contrast=[0.95,1.05], saturation=[0.95,1.05])(down_sampled_image)
        #     data_dict = {"ref":label_tensor, "low_res":down_sampled_image, "ref_ori":label_tensor_ori, "path": path}

        #     return image_tensor, data_dict
        if not self.train:
            return img, cond, index, fn
        else:
            return img, cond, index
        
    

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.cache is not None:
            return len(self.cache['img'])
        else:
            return len(self.filenames)
    


class MVTV(torch.utils.data.Dataset):
    """A dataset class for paired image dataset."""

    def __init__(self, dataroot, train=True, img_size=256, random_crop=False, random_flip=True, random_transform=True):
        """Initialize this dataset class.
        Parameters:
            dataroot (str) -- path to the root of the dataset
            train (bool) -- if True, create the dataset from the train set, otherwise from the val set
            img_size (int) -- the size to which the images are resized
            random_crop (bool) -- whether to randomly crop the images
            random_flip (bool) -- whether to randomly flip the images
        """
        super().__init__()
        self.image_root = os.path.join(dataroot, 'train' if train else 'val')
        self.crop_size = img_size
        self.resize_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train
        self.random_transform = random_transform

        # Filenames for Infrared and Visible images
        self.infrared_root = os.path.join(self.image_root, 'Infrared')
        self.visible_root = os.path.join(self.image_root, 'Visible')
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.infrared_root) if not f.endswith('.pth')]

        # Define transformations
        self.resize_transform = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains infrared, visible, infrared_paths and visible_paths
            infrared (tensor) - - an image in the infrared domain
            visible (tensor) - - its corresponding image in the visible domain
            infrared_paths (str) - - image paths
            visible_paths (str) - - image paths (same as infrared_paths)
        """
        fn = self.filenames[index]
        infrared_path = os.path.join(self.infrared_root, fn + '.JPG')
        visible_path = os.path.join(self.visible_root, fn + '.JPG')

        infrared_image = Image.open(infrared_path).convert("RGB")
        visible_image = Image.open(visible_path).convert("RGB")

        # Apply the same common transform to both infrared and visible images
        infrared_image = self.resize_transform(infrared_image)
        visible_image = self.resize_transform(visible_image)

        if self.train and self.random_transform:
            # Apply the same random transformations to both images
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            infrared_image = self._apply_transforms(infrared_image)
            random.seed(seed)
            torch.manual_seed(seed)
            visible_image = self._apply_transforms(visible_image)

        # Convert PIL images to tensors
        infrared_image = self.to_tensor(infrared_image)
        visible_image = self.to_tensor(visible_image)

        # return {'infrared': infrared_image, 'visible': visible_image, 'infrared_paths': infrared_path, 'visible_paths': visible_path}
        if not self.train:
            return infrared_image, visible_image, index, fn
        else: 
            return infrared_image, visible_image, index

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.filenames)

    def _apply_transforms(self, image):
        """Apply random transformations to the image."""
        if self.random_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)

        if self.random_crop:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            image = transforms.functional.crop(image, i, j, h, w)

        image = transforms.functional.rotate(image, angle=random.uniform(-10, 10))
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        image = color_jitter(image)

        return image