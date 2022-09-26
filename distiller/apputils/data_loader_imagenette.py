import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def download_dataset(data_dir):
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    dataset_name = "imagenette2-160"

    download_url(dataset_url, data_dir) # '.' here denotes the folder we are in, just as when we working on the terminal

    # Extract from archive
    with tarfile.open(dataset_name + ".tgz", 'r:gz') as tar:
        tar.extractall(path=f'{data_dir}/data')

    images_dir = f'{data_dir}/data/'+ dataset_name
    return images_dir
    
def load_imagenette(root, train = True, download = True, transform = [transforms.Resize((64,64)),transforms.ToTensor()]):
    if (download == True):
        download_dataset(root)
    if (train == True):
        dataset = ImageFolder(root+'/train', transform=transform)
    else:
        dataset = ImageFolder(root+'/val', transform=transform)

    dataset.classes = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    return dataset

# print(os.listdir(data_dir))
# classes = os.listdir(data_dir + "/train")
# print(classes)

# First_files = os.listdir(data_dir + "/train/" + classes[0])
# print('No. of training examples for airplanes:', len(First_files))
# print(First_files[:5])

# Second_val_files = os.listdir(data_dir + "/val/" + classes[0])
# print("No. of test examples for ship:", len(Second_val_files))
# print(Second_val_files[:5])

# def load_imagenette(root, train = True, download = True, transform = [transforms.Resize((64,64)),transforms.ToTensor()]):
#     if (download == True):
#         download_dataset(root)
#     if (train == True):
#         dataset = ImageFolder(root+'/train', transform=transform)
#     else:
#         dataset = ImageFolder(root+'/val', transform=transform)

#     dataset.classes = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
#     return dataset


# transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
# #Check more about ImageFolder if you have time
# dataset = ImageFolder(data_dir+'/train', transform=transform)
# # According to our original dataset, we could change the classname of our dataset
# dataset.classes = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
# val_ds = ImageFolder(data_dir+'/val', transform=transform)
# # According to our original dataset, we could change the classname of our dataset
# val_ds.classes = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']