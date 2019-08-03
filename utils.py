from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import json
from pycocotools.coco import COCO
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    def __init__(self, posetrack_dir, annotations_dir, device):
        annotations_list = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
        '''
        process all video annotations and store in frame batches (i.e. tensor
        of shape batch_size x image_height x image_width x image_channels)
        '''
        self.videos = []
        self.frame_counts = []
        self.video_keypoints = []

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        for annotations_file in tqdm(annotations_list):
            block_print()
            coco = COCO(os.path.join(annotations_dir, annotations_file))
            unblock_print()
            img_ids = coco.getImgIds()
            imgs = coco.loadImgs(img_ids)

            posetrack_images = []
            for img in imgs:
                if not img["is_labeled"]:
                    pass
                else:
                    posetrack_images.append(img)

            posetrack_images = posetrack_images[:10]

            video_frames = []
            frame_keypoints = []
            for image_idx, selected_im in enumerate(posetrack_images):
                # video_frame = Image.open(os.path.join(posetrack_dir, selected_im['file_name']))
                # video_frame = self.transform(video_frame)
                video_frame = os.path.join(posetrack_dir, selected_im['file_name'])
                video_frames.append(video_frame)
                ann_ids = coco.getAnnIds(imgIds=selected_im['id'])
                anns = coco.loadAnns(ann_ids)
                people_keypoints = []
                for ann in anns:
                    if 'keypoints' in ann and type(ann['keypoints']) == list:
                        kp = np.array(ann['keypoints'])
                        x = kp[0::3]
                        y = kp[1::3]
                        keypoints = torch.from_numpy(np.stack((x, y), axis=-1)).float()
                        people_keypoints.append(keypoints)

                frame_keypoints.append(torch.stack(people_keypoints, dim=0))

            self.videos.append(video_frames)
            self.frame_counts.append(len(video_frames))
            self.video_keypoints.append(frame_keypoints)
        self.device = device

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_keypoints = self.video_keypoints[idx]
        video = self.videos[idx]
        
        video_frames = []
        for frame in video:
            video_frames.append(self.transform(Image.open(frame)).to(self.device))
        
        return video_frames, video_keypoints

def block_print():
    sys.stdout = open(os.devnull, 'w')

def unblock_print():
    sys.stdout = sys.__stdout__

def collate_fn(data):
    batch_videos = [row[0] for row in data]
    #batch_videos = torch.stack(batch_videos, dim=0)
    batch_videos_keypoints = [row[1] for row in data]

    return batch_videos, batch_videos_keypoints

def load_train(batch_size=1, num_workers=0, device="cpu"):
    posetrack_dir = "./data/posetrack/"
    posetrack_annotations_dir = "./data/posetrack/posetrack_data/annotations"
    
    print("[I] Loading PoseTrack training set...")
    posetrack_train_set =  PoseTrackDataset(posetrack_dir, os.path.join(posetrack_annotations_dir, "train"), device=device)
    posetrack_train_dataloader = DataLoader(posetrack_train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    
    print("[II] Loading PoseTrack validation set...")
    posetrack_valid_set =  PoseTrackDataset(posetrack_dir, os.path.join(posetrack_annotations_dir, "val"), device=device)
    posetrack_valid_dataloader = DataLoader(posetrack_valid_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)

    return posetrack_train_dataloader, posetrack_valid_dataloader

def load_test(batch_size=1, num_workers=4, device="cpu"):
    posetrack_dir = "./data/posetrack/"
    posetrack_annotations_dir = "./data/posetrack/posetrack_data/annotations"
    
    print("[I] Loading PoseTrack testing set...")
    posetrack_test_set =  PoseTrackDataset(posetrack_dir, os.path.join(posetrack_annotations_dir, "test"), device=device)
    posetrack_test_dataloader = DataLoader(posetrack_test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    
    return posetrack_test_dataloader
