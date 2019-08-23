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
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io as io

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
    def __init__(self, posetrack_dir, coco_dir, annotations_dir, device):
        '''
        coco for instances
        '''
        block_print()
        coco_instances = COCO(coco_dir)
        unblock_print()
        annotations_list = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
        '''
        process all video annotations and store in frame batches (i.e. tensor
        of shape batch_size x image_height x image_width x image_channels)
        '''
        self.videos = []
        self.frame_counts = []
        self.video_keypoints = []
        self.video_scales = []
        self.video_bboxes = []

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

            posetrack_images = posetrack_images[0:10]

            video_frames = []
            frame_keypoints = []
            frame_scales = []
            frame_bboxes = []
            for image_idx, selected_im in enumerate(posetrack_images):
                video_frame = os.path.join(posetrack_dir, selected_im['file_name'])
                video_frames.append(video_frame)
                ann_ids = coco.getAnnIds(imgIds=selected_im['id'])
                anns = coco.loadAnns(ann_ids)
                people_keypoints = []
                people_scales = []
                people_bboxes = []
                for ann in anns:
                    if 'bbox' in ann and 'bbox_head' in ann and 'keypoints' in ann and type(ann['keypoints']) == list:
                        head_bbox = ann['bbox_head']
                        bbox = torch.FloatTensor(ann['bbox']).to(device)
                        kp = np.array(ann['keypoints'])
                        x = kp[0::3]
                        y = kp[1::3]
                        v = kp[2::3]
                        s = math.sqrt(head_bbox[2] * head_bbox[3])
                        people_scales.append(s)
                        people_bboxes.append(bbox)
                        keypoints = torch.from_numpy(np.stack((x, y, v), axis=-1)).float()
                        people_keypoints.append(keypoints)
                frame_keypoints.append(torch.stack(people_keypoints, dim=0))
                frame_scales.append(people_scales)
                frame_bboxes.append(people_bboxes)

            self.videos.append(video_frames)
            self.frame_counts.append(len(video_frames))
            self.video_keypoints.append(frame_keypoints)
            self.video_scales.append(frame_scales)
            self.video_bboxes.append(frame_bboxes)
        self.device = device

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        orig_video_keypoints = self.video_keypoints[idx]
        orig_video_scales = self.video_scales[idx]
        orig_video_bboxes = self.video_bboxes[idx]
        video = self.videos[idx]
        
        video_frames = []
        video_keypoints = []
        video_scales = []
        video_bboxes = []
        base_width = 736
        for frame, kps, scales, bboxes in zip(video, orig_video_keypoints, orig_video_scales, orig_video_bboxes):
            resized_image = Image.open(frame)
            orig_width = resized_image.size[0]
            orig_height = resized_image.size[1]
            wp = (base_width / float(orig_width))
            im_height = int((float(orig_height) * float(wp)))
            hp = (im_height / float(orig_height))
            resized_image = resized_image.resize((base_width, im_height), Image.ANTIALIAS)
            video_frame = self.transform(resized_image).to(self.device)
            kps[:,:,0] = wp * kps[:,:,0]
            kps[:,:,1] = hp * kps[:,:,1]
            video_keypoints.append(kps)
            curr_scales = []
            for i in range(len(scales)):
                curr_scales.append(math.sqrt(hp) * math.sqrt(wp) * scales[i])
                bboxes[i][0] = wp * bboxes[i][0]
                bboxes[i][1] = hp * bboxes[i][1]
                bboxes[i][2] = wp * bboxes[i][2]
                bboxes[i][3] = hp * bboxes[i][3]
            video_bboxes.append(bboxes)
            video_scales.append(curr_scales)
            video_frames.append(video_frame)
        
        return video_frames, video_keypoints, video_scales, video_bboxes

def block_print():
    sys.stdout = open(os.devnull, 'w')

def unblock_print():
    sys.stdout = sys.__stdout__

def collate_fn(data):
    batch_videos = [row[0] for row in data]
    #batch_videos = torch.stack(batch_videos, dim=0)
    batch_videos_keypoints = [row[1] for row in data]
    batch_videos_keypoints = list(map(list, zip(*batch_videos_keypoints)))
    batch_videos_scales = [row[2] for row in data]
    batch_videos_bboxes = [row[3] for row in data]
    return batch_videos, batch_videos_keypoints, batch_videos_scales, batch_videos_bboxes

def load_train(batch_size=1, num_workers=0, device="cpu"):
    posetrack_dir = "./data/posetrack/"
    coco_dir = "./data/coco/annotations/instances_train2017.json"
    posetrack_annotations_dir = "./data/posetrack/posetrack_data/annotations"
    
    print("[I] Loading PoseTrack training set...")
    posetrack_train_set =  PoseTrackDataset(posetrack_dir, coco_dir, os.path.join(posetrack_annotations_dir, "train"), device=device)
    posetrack_train_dataloader = DataLoader(posetrack_train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    
    print("[II] Loading PoseTrack validation set...")
    coco_dir = "./data/coco/annotations/instances_val2017.json"
    posetrack_valid_set =  PoseTrackDataset(posetrack_dir, coco_dir, os.path.join(posetrack_annotations_dir, "val"), device=device)
    posetrack_valid_dataloader = DataLoader(posetrack_valid_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)

    return posetrack_train_dataloader, posetrack_valid_dataloader

def load_test(batch_size=1, num_workers=4, device="cpu"):
    posetrack_dir = "./data/posetrack/"
    coco_dir = "./data/coco/"
    posetrack_annotations_dir = "./data/posetrack/posetrack_data/annotations"
    
    print("[I] Loading PoseTrack testing set...")
    posetrack_test_set =  PoseTrackDataset(posetrack_dir, coco_dir, os.path.join(posetrack_annotations_dir, "test"), device=device)
    posetrack_test_dataloader = DataLoader(posetrack_test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    
    return posetrack_test_dataloader
