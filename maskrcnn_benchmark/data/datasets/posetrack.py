import torch
import torchvision

from pycocotools.coco import COCO
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

import os
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if boxes are not labeled for all objects in annotation, the annotation is invalid
    bbox_valid = all("bbox" in obj for obj in anno)
    if not bbox_valid:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

class PoseTrackDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_dir, root_posetrack, remove_images_without_annotations, transforms=None, is_train=True):
        super(PoseTrackDataset, self).__init__()
        self.root_posetrack = root_posetrack
        annotations_list = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
        # per video annotations
        self.categories = {}
        self.json_category_id_to_contiguous_id = {}
        self.contiguous_category_id_to_json_id = {}
        self.id_to_video_map = {}
        self.id_to_annotation_file_map = {}
        start_id = 0
        self.video_img_ids = []
        self.video_annotations = []
        self.coco_objs = []
        self.del_idxs = []
        for annotations_file in tqdm(annotations_list):
            block_print()
            coco = COCO(os.path.join(annotations_dir, annotations_file))
            unblock_print()
            img_ids = coco.getImgIds()
            
            # posetrack valid image ids for videos
            posetrack_image_ids = []
            posetrack_annotations = []
            for img_id in img_ids:
                ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = coco.loadAnns(ann_ids)
                if not remove_images_without_annotations:
                    posetrack_image_ids.append(img_id)
                    posetrack_annotations.append(anno)
                else:
                    if has_valid_annotation(anno):
                        posetrack_image_ids.append(img_id)
                        posetrack_annotations.append(anno)
                    else:
                        continue
            #seq_len = len(posetrack_image_ids)
            # just 5 frames of a video are used currently to train
            if is_train:
                seq_len = 5
            else:
                seq_len = 15
            posetrack_image_ids = posetrack_image_ids[0:seq_len]
            posetrack_annotations = posetrack_annotations[0:seq_len]

            if len(posetrack_image_ids) > 0:

                self.video_img_ids.append(posetrack_image_ids)
                self.video_annotations.append(posetrack_annotations)
                # store coco for later extraction
                self.coco_objs.append(coco)

                # update categories dict
                self.categories.update({cat['id']: cat['name'] for cat in coco.cats.values()})
                self.json_category_id_to_contiguous_id.update({
                    v: i + 1 for i, v in enumerate(coco.getCatIds())
                })
                self.contiguous_category_id_to_json_id.update({
                    v: k for k, v in self.json_category_id_to_contiguous_id.items()
                })
                self.id_to_video_map.update({
                    start_id: posetrack_image_ids
                })
                self.id_to_annotation_file_map.update({
                    start_id: annotations_file
                })
                start_id = len(self.id_to_video_map)
                # print({
                #     start_id + k: v for k, v in enumerate(posetrack_image_ids)
                # })
                # self.id_to_video_map.update({
                #     start_id + k: v for k, v in enumerate(posetrack_image_ids)
                # })
                # start_id = list(self.id_to_video_map.keys())[-1] + 1
            
        self._transforms = transforms

    def __getitem__(self, idx):
        # filter crowd annotations
        images = []
        targets = []
        coco = self.coco_objs[idx]
        video_ids = self.video_img_ids[idx]
        video_frames = coco.loadImgs(video_ids)
        video_annotation = self.video_annotations[idx]
        for image, annotation in zip(video_frames, video_annotation):
            new_im = list()
            new_ann = list()
            use = True
            for obj in annotation:
                if "bbox" not in obj:
                    use = False
                    print("filtered annotation that was not fully labeled...")
                    self.del_idxs.append(idx)
                    break
            if use:
                image = Image.open(os.path.join(self.root_posetrack, image['file_name'])).convert("RGB")
                #annotation = [obj for obj in annotation if obj["iscrowd"] == 0]
                boxes = [obj["bbox"] for obj in annotation]
                boxes = torch.as_tensor(boxes).reshape(-1, 4) # guard against no boxes
                target = BoxList(boxes, image.size, mode="xywh").convert("xyxy")

                classes = [obj["category_id"] for obj in annotation]
                classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
                classes = torch.tensor(classes)
                target.add_field("labels", classes)

                if annotation and "keypoints" in annotation[0]:
                    keypoints = [obj["keypoints"] for obj in annotation]
                    keypoints = PersonKeypoints(keypoints, image.size)
                    target.add_field("keypoints", keypoints)
                
                #visualize(image, boxes, keypoints)
                
                target = target.clip_to_image(remove_empty=True)

                if self._transforms is not None:
                    image, target = self._transforms(image, target)

                images.append(image)
                targets.append(target)
        
        return images, targets, idx
    
    def __len__(self):
        return len(self.video_annotations)

    def get_img_info(self, index):
        img_id = self.id_to_video_map[index][0] # all image frames for a video should be similar
        img_data = self.coco_objs[index].imgs[img_id]
        img = Image.open(os.path.join('datasets/posetrack', img_data['file_name'])).convert("RGB")
        img_data['width'] = img.size[0]
        img_data['height'] = img.size[1]
        return img_data

def block_print():
    sys.stdout = open(os.devnull, 'w')

def unblock_print():
    sys.stdout = sys.__stdout__

def visualize(image, boxes=None, keypoints=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    if boxes is not None and keypoints is not None:
        for testing in zip(boxes, keypoints):
            box = testing[0]
            keypoint = testing[1].keypoints.squeeze(dim=1)
            x = keypoint[:,0]
            y = keypoint[:,1]
            v = keypoint[:,2]
            indices = (v > 0).nonzero()
            x = x[indices].tolist()
            y = y[indices].tolist()
            rect = patches.Rectangle((box[0], box[1]),box[2], box[3],linewidth=1,edgecolor='r',facecolor='none')
            plt.scatter(x, y)
            ax.add_patch(rect)
    plt.show()
    plt.close()
