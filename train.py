import torchvision
import torch
from model import MedPose
from utils import load_train
import argparse
import os
import time

DEVICES = [0, 1, 2]

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 5
batch_size = len(DEVICES) - 1
num_keypoints = 17

train_dataloader, valid_dataloader = load_train(batch_size=batch_size, device=device)
model = MedPose(window_size=window_size, num_keypoints=num_keypoints, device=device, gpus=DEVICES)

optimized_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
optimizer = torch.optim.Adam(optimized_params, lr=0.01)
estimation_criterion = torch.nn.MSELoss()
classification_criterion = torch.nn.CrossEntropyLoss()

def train(args):
    print("[III] Training MedPose...")
    print("Number of trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for epoch in range(args.epochs + 1):
        start_time = time.time()
        for train_idx, (batch_videos, batch_keypoints) in enumerate(train_dataloader):
            estimations, classifications = model(batch_videos)
            estimation_total_loss = 0
            classification_total_loss = 0
            for seq_idx in range(len(estimations)):
                for batch_idx in range(batch_size):
                    pose_estimations = estimations[seq_idx][batch_idx].to(device)
                    pose_estimations = pose_estimations.view(
                            pose_estimations.shape[0], 2, num_keypoints).permute(0, 2, 1)
                    classification = classifications[seq_idx][batch_idx].to(device)
                    keypoints = batch_keypoints[seq_idx][batch_idx].to(device)
                    # get tight bounding box from keypoints
                    gt_min_bounds, gt_max_bounds = find_bounds_from_keypoints(keypoints)
                    # get bounding box from pose estimations
                    pred_min_bounds, pred_max_bounds = find_bounds_from_keypoints(pose_estimations)
                    # generate labels to compute loss for regression and classification
                    keypoint_labels, classification_labels = generate_labels(pose_estimations, keypoints, 
                             pred_min_bounds, pred_max_bounds, gt_min_bounds, gt_max_bounds)
                    estimation_total_loss += estimation_criterion(pose_estimations, keypoint_labels)
                    classification_total_loss += classification_criterion(classification, classification_labels)
                    del pose_estimations
                    del classification
                    del keypoints
                    del keypoint_labels
                    del classification_labels
            # backpropogate gradients from loss functions and update weights
            optimizer.zero_grad()
            total_loss = estimation_total_loss + classification_total_loss
            total_loss.backward()
            optimizer.step()
            # write out current epoch and losses and delete memory consuming variables
            print("Epoch: {}/{}\tLoss: {}, {}".format(epoch, args.epochs, estimation_total_loss, classification_total_loss))
            del estimations
            del classifications
            del estimation_total_loss
            del classification_total_loss
        print("Time to complete epoch: {} seconds".format(time.time() - start_time))
        if epoch != 0 and epoch > 50 and epoch % 10 == 0:
            torch.save(model.state_dict(), "model_repository/model-{}.pth".format(epoch))

def generate_labels(pred_tensor, gt_tensor, pred_min_bounds, pred_max_bounds, gt_min_bounds, gt_max_bounds):
    matched_gt_tensor = pred_tensor.clone()
    classification_labels = torch.zeros(pred_tensor.shape[0]).long().to(device)
    for gt_region_idx in range(gt_tensor.shape[0]):
        max_sim_score = -1
        max_pred_idx = -1
        for pred_region_idx in range(pred_tensor.shape[0]):
            sim_score = iou_score(
                    pred_min_bounds[pred_region_idx], pred_max_bounds[pred_region_idx],
                    gt_min_bounds[gt_region_idx], gt_max_bounds[gt_region_idx])
            if sim_score > max_sim_score:
                max_sim_score = sim_score
                max_pred_idx = pred_region_idx
        matched_gt_tensor[max_pred_idx] = gt_tensor[gt_region_idx]
        # do we need to match by some thresh?
        #if max_sim_score > thresh:
        #    classification_labels[max_pred_idx] = 1
        classification_labels[max_pred_idx] = 1
    return matched_gt_tensor, classification_labels

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

def find_bounds_from_keypoints(keypoints):
    min_bounds, _ = torch.min(keypoints, dim=1)
    max_bounds, _ = torch.max(keypoints, dim=1)
    return min_bounds, max_bounds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1)
    args = parser.parse_args()
    # check if model repository exists otherwise create it
    if not os.path.exists("model_repository"):
        os.mkdir("model_repository")
    train(args)

