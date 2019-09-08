import torch
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from model import MedPose
from utils import load_train
import numpy as np
import math
import argparse
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def test_model(args):

    if not torch.cuda.is_available():
        sys.exit("Need CUDA to test MedPose")

    DEVICES = [i for i in range(args.gpus)]

    window_size = 1
    batch_size = args.batch_per_gpu * (len(DEVICES))
    num_keypoints = 17

    # loading train posetrack dataset 
    train_dataloader, valid_dataloader = load_train(batch_size=batch_size, device=device)

    model = MedPose(window_size=window_size, num_keypoints=num_keypoints, num_rpn_props=300, stack_layers=args.stack_layers, device=device, gpus=DEVICES)
    model.load_state_dict(torch.load(args.model))
    model.base.to(DEVICES[0])
    model.encoder.to(DEVICES[1])
    model.pose_cl.to(DEVICES[1])
    model.pose_regress.to(DEVICES[1])
    #model.decoder.to(DEVICES[1])

    model.encoder = torch.nn.DataParallel(model.encoder, device_ids=DEVICES[1:])
    model.pose_cl = torch.nn.DataParallel(model.pose_cl, device_ids=DEVICES[1:])
    model.pose_regress = torch.nn.DataParallel(model.pose_regress, device_ids=DEVICES[1:])
    #model.decoder = torch.nn.DataParallel(model.decoder, device_ids=DEVICES[1:])
    

    print("[III] Testing MedPose...")
    model.eval()
    assert(sum(p.numel() for p in model.parameters() if p.requires_grad) == 0)
    
    running_counts = []
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        for train_idx, (batch_videos, batch_keypoints, batch_scales, batch_bboxes) in enumerate(train_dataloader):
            estimations, classifications, proposals = model(batch_videos)
            classification_accs = []
            gt_accs = []
            for seq_idx in range(len(estimations)):
                for batch_idx in range(estimations[seq_idx].shape[0]):
                    pose_estimations = estimations[seq_idx][batch_idx].to(device)
                    pose_estimations = pose_estimations.view(
                            pose_estimations.shape[0], 2, num_keypoints).permute(0, 2, 1)
                    classification = classifications[seq_idx][batch_idx].to(device)
                    region_props = proposals[seq_idx][batch_idx].to(device)
                    keypoints = batch_keypoints[seq_idx][batch_idx].to(device)
                    visibility_labels = keypoints[:,:,2]
                    keypoints = keypoints[:,:,:2]
                    scales = batch_scales[batch_idx][seq_idx]
                    bboxes = batch_bboxes[batch_idx][seq_idx]
                    '''
                    get labels for ground truth
                    '''
                    keypoint_preds, classification_labels = refine_output_and_gt_labels(region_props, bboxes, pose_estimations, keypoints)
                    threshold = 0.75
                    ap, _ = compute_ap(keypoint_preds, keypoints, visibility_labels, scales, threshold, running_counts)
                    # visualizations
                    gt_indices = (classification_labels == 1).nonzero()
                    #pred_indices = (pred_labels == 1).nonzero()
                    visualize_predictions_and_gts(batch_videos[batch_idx][seq_idx], keypoints, keypoint_preds, visibility_labels, bboxes, region_props[gt_indices], region_props)
                    #visualize_predictions_and_gts(batch_videos[batch_idx][seq_idx], keypoints, keypoint_preds, visibility_labels, bboxes, region_props[pred_indices])
                    #visualize_predictions_and_gts(batch_videos[batch_idx][seq_idx], keypoints, keypoint_preds, visibility_labels, region_props[gt_indices], region_props[pred_indices])

def compute_ap(pred_tensor, gt_tensor, visibility_tensor, scales, threshold, running_counts=[]):
    object_oks_scores = torch.zeros(visibility_tensor.shape[0]).float()
    for object_idx in range(visibility_tensor.shape[0]):
        labeled_keypoints = torch.sum(visibility_tensor[object_idx] > 0)
        pred_keypoints = pred_tensor[object_idx]
        gt_keypoints = gt_tensor[object_idx]
        pred_keypoints = pred_keypoints[(visibility_tensor[object_idx] > 0).nonzero().squeeze(dim=1)]
        gt_keypoints = gt_keypoints[(visibility_tensor[object_idx] > 0).nonzero().squeeze(dim=1)]
        d = torch.sqrt(torch.sum(torch.pow((pred_keypoints - gt_keypoints), 2), dim=1))
        s = scales[object_idx]
        k = 2 * (d / float(s)) # 2 standard deviations fall off
        oks = torch.sum(torch.exp(torch.neg(torch.pow(d, 2)) / (2 * (k ** 2)))) / labeled_keypoints.float()
        object_oks_scores[object_idx] = oks
    # compute mean average precision of oks scores (similar to evaluation server)
    true_pos = torch.sum(object_oks_scores > threshold)
    false_pos = torch.sum(object_oks_scores <= threshold)
    if len(running_counts) == 0:
        running_counts = [true_pos, false_pos]
    else:
        running_counts += [true_pos, false_pos]
    ap = float(running_counts[0]) / sum(running_counts)
    return ap, running_counts

def _visualize_region_props(image, region_props):
    '''
    plot setup
    '''
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for region_idx in range(len(region_props)):
        bbox = torch.round(region_props[region_idx])
        x1 = bbox[0].item()
        y1 = bbox[1].item()
        x2 = bbox[2].item()
        y2 = bbox[3].item()
        rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()
    plt.close()

def visualize_predictions_and_gts(image, gt_keypoints, pred_keypoints, visibility_labels, gt_boxes, pred_boxes, region_props):
    # plot image
    image = image.permute(1, 2, 0).cpu().numpy()
    #_visualize_region_props(image, region_props)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for person_idx in range(gt_keypoints.shape[0]):
        gt_points = gt_keypoints[person_idx]
        if person_idx < pred_keypoints.shape[0]:
            pred_points = pred_keypoints[person_idx]
        else:
            pred_points = None
        gt_bbox = torch.round(gt_boxes[person_idx])
        pred_bbox = None
        vis = (visibility_labels[person_idx] > 0).nonzero().squeeze(dim=1)
        gt_points = gt_points[vis]
        if pred_points is not None and pred_boxes.shape[1] > 0:
            pred_bbox = torch.round(pred_boxes[person_idx].squeeze(dim=0))
        # plot keypoints
        gt_x = gt_points[:,0].tolist()
        gt_y = gt_points[:,1].tolist()
        if pred_points is not None:
            pred_x = pred_points[:,0].tolist()
            pred_y = pred_points[:,1].tolist()
        # plot green points for gt_keypoints
        ax.scatter(x=gt_x, y=gt_y, c='g', s=10)
        # plot blue points for pred_keypoints
        if pred_points is not None:
            ax.scatter(x=pred_x, y=pred_y, c='r', s=10)
        # plot bboxes
        gt_x1 = gt_bbox[0].item()
        gt_y1 = gt_bbox[1].item()
        gt_w = gt_bbox[2].item()
        gt_h = gt_bbox[3].item()
        if pred_bbox is not None:
            pred_x1 = pred_bbox[0].item()
            pred_y1 = pred_bbox[1].item()
            pred_x2 = pred_bbox[2].item()
            pred_y2 = pred_bbox[3].item()
        # plot green rects for gt objects
        gt_rect = patches.Rectangle((gt_x1, gt_y1), gt_w, gt_h, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(gt_rect)
        # plot red rects for pred objects
        if pred_bbox is not None:
            pred_rect = patches.Rectangle((pred_x1, pred_y1), (pred_x2 - pred_x1), (pred_y2 - pred_y1), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(pred_rect)
    plt.show()
    plt.close()

def refine_output_and_gt_labels(region_props, gt_bboxes, pred_tensor, gt_tensor):
    keypoint_preds = gt_tensor.clone()
    classification_labels = torch.zeros(region_props.shape[0]).long().to(device)
    used = set()
    for gt_idx, gt_box in enumerate(gt_bboxes):
        max_sim_score = -1
        max_pred_idx = -1
        for region_idx, region_prop in enumerate(region_props):
            t1_min_bounds = torch.stack((gt_box[0], gt_box[1]), dim=0)
            t1_max_bounds = torch.stack((gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]), dim=0)
            t2_min_bounds = torch.stack((region_prop[0], region_prop[1]), dim=0)
            t2_max_bounds = torch.stack((region_prop[2], region_prop[3]), dim=0)
            sim_score = iou_score(t1_min_bounds, t1_max_bounds, t2_min_bounds, t2_max_bounds)
            if sim_score > max_sim_score and region_idx not in used:
                max_sim_score = sim_score
                max_pred_idx = region_idx
        classification_labels[max_pred_idx] = 1
        keypoint_preds[gt_idx] = pred_tensor[max_pred_idx]
        used.add(max_pred_idx)
    return keypoint_preds, classification_labels

def iou_score(t1_min_bounds, t1_max_bounds, t2_min_bounds, t2_max_bounds):
    # only one bounding represented by each tensor
    assert(len(t1_min_bounds.shape) == 1)
    assert(len(t2_min_bounds.shape) == 1)
    assert(len(t1_max_bounds.shape) == 1)
    assert(len(t2_max_bounds.shape) == 1)
    t_min_bounds = torch.stack((t1_min_bounds.float(), t2_min_bounds.float()), dim=0)
    t_max_bounds = torch.stack((t1_max_bounds.float(), t2_max_bounds.float()), dim=0)
    #intersection computation
    intersection_mins = torch.neg(torch.max(t_min_bounds, dim=0)[0])
    intersection_maxs = torch.min(t_max_bounds, dim=0)[0]
    intersection_stack = torch.stack((intersection_maxs, intersection_mins), dim=0)
    intersection_vals = torch.sum(intersection_stack, dim=0)
    intersection_area = torch.mul(intersection_vals[0], intersection_vals[1])
    #union computation
    union_mins = torch.neg(torch.min(t_min_bounds, dim=0)[0])
    union_maxs = torch.max(t_max_bounds, dim=0)[0]
    union_stack = torch.stack((union_maxs, union_mins), dim=0)
    union_vals = torch.sum(union_stack, dim=0)
    union_area = torch.mul(union_vals[0], union_vals[1])
    return torch.div(intersection_area, union_area.float())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--gpus", default=2, type=int)
    parser.add_argument("--batch_per_gpu", default=1, type=int)
    parser.add_argument("--stack_layers", default=4, type=int)
    args = parser.parse_args()
    if args.model:
        test_model(args)
