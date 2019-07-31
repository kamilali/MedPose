from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import cv2
import os
import json
import pprint

"""
Dataset class that reads the MPII dataset
"""
class MPIIDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class PoseTrackDataset(Dataset):
    def __init__(self, images_dir, annotations_dir):
        annotations_list = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]

        for annotations_file in annotations_list:
            print(annotations_file)
            with open(os.path.join(annotations_dir, annotations_file), 'r') as json_file:
                curr_annotations = json.load(json_file)
            pprint.pprint(curr_annotations)
            exit()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def load():
    posetrack_images_dir = "./data/posetrack/images"
    posetrack_annotations_dir = "./data/posetrack/posetrack_data/annotations"
    
    posetrack_train_set =  PoseTrackDataset(os.path.join(posetrack_images_dir, "train"), os.path.join(posetrack_annotations_dir, "train"))
