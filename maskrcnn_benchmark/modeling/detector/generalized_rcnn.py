# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = to_image_list(images)
        vid_shape = None
        if len(images.tensors.shape) > 4:
            vid_shape = images.tensors.shape[:2] # just get batch and sequence dimensions
            images.tensors = images.tensors.view(-1, images.tensors.shape[2], images.tensors.shape[3], images.tensors.shape[4])
        # for img_idx in range(images.tensors.shape[0]):
        #     img = images.tensors[img_idx]
        #     visualize(img)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        #features = [feature.view(*vid_shape, feature.shape[1], feature.shape[2], feature.shape[3]) for feature in features]
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, vid_shape)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

def visualize(image, boxes=None, keypoints=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1,2,0).detach().cpu().numpy())
    if boxes is not None or keypoints is not None:
        if boxes is None:
            enumerable = keypoints
        elif keypoints is None:
            enumerable = boxes
        else:
            enumerable = zip(boxes, keypoints)
        for testing in enumerable:
            if boxes is not None and keypoints is not None:
                box = testing[0]
                if not isinstance(testing[1], list):
                    keypoint = testing[1].keypoints.squeeze(dim=1)
                else:
                    keypoint = testing[1]
            elif boxes is not None:
                box = testing
            else:
                keypoint = testing
            if keypoints is not None:
                x = keypoint[0::3]
                y = keypoint[1::3]
                v = keypoint[2::3]
                temp_x, temp_y = [], []
                for l in range(len(v)):
                    if v[l] > 0:
                        temp_x.append(x[l])
                        temp_y.append(y[l])
                x = temp_x
                y = temp_y
            if boxes is not None:
                rect = patches.Rectangle((box[0], box[1]),box[2], box[3],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            if keypoints is not None:
                plt.scatter(x, y)
    plt.show()
    plt.close()
