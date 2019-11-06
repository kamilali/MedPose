# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.data.datasets.evaluation.poseval.py import evaluate_simple
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList

import numpy as np
from scipy.optimize import linear_sum_assignment

import copy
import shutil
import json

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MAX_TRACK_IDS = 999
FIRT_TRACK_ID = 0

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        if isinstance(images, list) and len(images) == 0:
            if cfg.POSETRACK_EVAL:
                continue
            else:
                break
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            if not cfg.POSETRACK_EVAL:
                output = [o.to(cpu_device) for o in output]
            else:
                outputs = output
                for idx, output in enumerate(outputs):
                    output = [o.to(cpu_device) for o in output]
                    outputs[idx] = output
        if not cfg.POSETRACK_EVAL:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
        else:
            item = {}
            for output in outputs:
                for img_id, result in zip(image_ids, output):
                    if img_id in item:
                        item[img_id].append(result)
                    else:
                        item[img_id] = [result]
            results_dict.update(item)
            # images_seq = images.tensors[0]
            # for idx in range(images_seq.shape[0]):
            #     curr_box = results_dict[0][idx]
            #     visualize(images_seq[idx], curr_box.bbox)
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return
    
    if cfg.POSETRACK_EVAL:
        # clear predictions from prior run
        shutil.rmtree(output_folder)
        os.mkdir(output_folder)
        annotation_dir = os.path.join(DatasetCatalog.DATA_DIR, DatasetCatalog.DATASETS[dataset_name]['ann_dir'])
        #print(len(predictions))
        #print(len(dataset.id_to_video_map))
        #print(len(dataset.id_to_annotation_file_map))
        #predictions_dict = {}
        for vid_id, prediction in enumerate(predictions):
            original_id = dataset.id_to_video_map[vid_id]
            annotation_file = dataset.id_to_annotation_file_map[vid_id]
            posetrack_results = []
            #print(original_id)
            # get annotation file and create mapping for indexing faster
            annotation_file = os.path.join(annotation_dir, annotation_file)
            annotation_json = json.load(open(annotation_file))
            # initialize prediction json with empty annotations list
            predicted_json = copy.deepcopy(annotation_json)
            #predicted_json["annotations"] = []
            predicted_json["annotations"] = [None for i in range(len(annotation_json["annotations"]))]
            annotation_json = annotation_json["annotations"]
            annotation_mappings = {}
            for i in range(len(annotation_json)):
                image_id = annotation_json[i]['image_id']
                ann_key = str(image_id)
                if ann_key not in annotation_mappings:
                    annotation_mappings[ann_key] = []
                annotation_mappings[ann_key].append(i)
            #print(annotation_mappings.keys())
            # preprocess images using video info
            vid_info = dataset.get_img_info(vid_id)

            # tracking
            next_track_id = FIRT_TRACK_ID
            video_tracks = []
            for frame_id, (im_id, im_prediction) in enumerate(zip(original_id, prediction)):
                frame_tracks = []
                curr_boxes = im_prediction.bbox.numpy()
                curr_poses = im_prediction.get_field('keypoints').keypoints.numpy()
                if frame_id == 0:
                    matches = -np.ones((curr_boxes.shape[0],))
                else:
                    prev_boxes = prediction[frame_id - 1].bbox.numpy()
                    prev_poses = prediction[frame_id - 1].get_field('keypoints').keypoints.numpy()
                    matches = _compute_matches(prediction[frame_id - 1], im_prediction, prev_boxes, curr_boxes, prev_poses, curr_poses)
                prev_tracks = video_tracks[frame_id - 1] if frame_id > 0 else None
                for m in matches:
                    # didnt match to any
                    if m == -1:
                        frame_tracks.append(next_track_id)
                        next_track_id += 1
                        if next_track_id > MAX_TRACK_IDS:
                            print("exceeded max track ids")
                            next_track_id %= MAX_TRACK_IDS
                    else:
                        frame_tracks.append(prev_tracks[m])
                video_tracks.append(frame_tracks)

            for im_id, im_prediction, frame_tracks in zip(original_id, prediction, video_tracks):
                # ensure that image ids are unique
                #assert im_id not in predictions_dict, "videos have images with overlapping ids"
                vid_width = vid_info['width']
                vid_height = vid_info['height']
                im_prediction = im_prediction.resize((vid_width, vid_height))
                im_prediction = im_prediction.convert('xywh')
                #predictions_dict[im_id] = im_prediction

                #print(im_id, im_prediction.fields(), "image prediction fields")
                boxes = im_prediction.bbox.tolist()
                keypoints = im_prediction.get_field('keypoints')
                if keypoints.keypoints.shape[0] != 0:
                    keypoint_scores = keypoints.get_field('logits').view(keypoints.keypoints.shape[0], -1)
                    keypoints = keypoints.resize((vid_width, vid_height))
                    keypoints = keypoints.keypoints.view(keypoints.keypoints.shape[0], -1)
                    #print(keypoints.keypoints.tolist())
                    scores = im_prediction.get_field('scores').tolist()
                    labels = im_prediction.get_field('labels').tolist()

                    # match per region and create annotation
                    #print(len(keypoints.keypoints))
                    ann_key = str(im_id)
                    # check if annotation exists for current image
                    if ann_key in annotation_mappings:
                        a_idxs = annotation_mappings[ann_key]
                        for a_idx in a_idxs:
                            gt_ann = annotation_json[a_idx]
                            # print(gt_ann.keys(), "ground truth annotation keys")
                            gt_box = BoxList([gt_ann['bbox']], (vid_width, vid_height), mode='xywh')
                            box_ious = boxlist_iou(im_prediction, gt_box)
                            # ensure that a corresponding box exists
                            if box_ious.shape[0] != 0:
                                match_idx = box_ious.max(0)[1].item()
                                pred_ann = {}
                                pred_ann['bbox'] = boxes[match_idx]
                                pred_ann['keypoints'] = keypoints[match_idx].tolist()
                                pred_ann['track_id'] = frame_tracks[match_idx]
                                pred_ann['image_id'] = im_id
                                pred_ann['scores'] = keypoint_scores[match_idx].tolist()
                                predicted_json["annotations"][a_idx] = pred_ann
                    else:
                        print("annotation does not exist for current image... no predictions")
                else:
                    print("no predictions for: ", im_id)
            # write out prediction file for current video 
            output_file = os.path.join(output_folder, dataset.id_to_annotation_file_map[vid_id])
            print("Saving current video predictions to", output_file)
            with open(output_file, 'w') as f:
                json.dump(predicted_json, f)

        # pose track evaluation handled seperately
        annotation_dir = os.path.join(annotation_dir, '')
        output_folder = os.path.join(output_folder, '')
        evaluate_simple.evaluate(annotation_dir, output_folder, True, False, False)
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        posetrack_eval=cfg.POSETRACK_EVAL,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)

def _compute_pairwise_iou(a, b):
    """
    a, b (np.ndarray) of shape Nx4T and Mx4T.
    The output is NxM, for each combination of boxes.
    """
    ious = boxlist_iou(a, b)
    return ious

def _compute_distance_matrix(
    prev_json_data, prev_boxes, prev_poses,
    cur_json_data, cur_boxes, cur_poses,
    cost_types, cost_weights
):
    assert(len(cost_weights) == len(cost_types))
    all_Cs = []
    for cost_type, cost_weight in zip(cost_types, cost_weights):
        if cost_weight == 0:
            continue
        if cost_type == 'bbox-overlap':
            all_Cs.append((1 - _compute_pairwise_iou(prev_json_data, cur_json_data)))
        else:
            raise NotImplementedError('Unknown cost type {}'.format(cost_type))
        all_Cs[-1] *= cost_weight
    return np.sum(np.stack(all_Cs, axis=0), axis=0)

def _compute_matches(prev_frame_data, cur_frame_data, prev_boxes, cur_boxes,
                     prev_poses, cur_poses, cost_types=('bbox-overlap',), cost_weights=(1.0,), bipart_match_algo="hungarian"):
    """
    C (cost matrix): num_prev_boxes x num_current_boxes
    Optionally input the cost matrix, in which case you can input dummy values
    for the boxes and poses
    Returns:
        matches: A 1D np.ndarray with as many elements as boxes in current
        frame (cur_boxes). For each, there is an integer to index the previous
        frame box that it matches to, or -1 if it doesnot match to any previous
        box.
    """
    # matches structure keeps track of which of the current boxes matches to
    # which box in the previous frame. If any idx remains -1, it will be set
    # as a new track.
    nboxes = cur_boxes.shape[0]
    matches = -np.ones((nboxes,), dtype=np.int32)
    C = _compute_distance_matrix(
        prev_frame_data, prev_boxes, prev_poses,
        cur_frame_data, cur_boxes, cur_poses,
        cost_types=cost_types,
        cost_weights=cost_weights)
    if bipart_match_algo == 'hungarian':
        prev_inds, next_inds = linear_sum_assignment(C)
    else:
        raise NotImplementedError('Unknown matching algo: {}'.format(
            bipart_match_algo))
    assert(len(prev_inds) == len(next_inds))
    for i in range(len(prev_inds)):
        matches[next_inds[i]] = prev_inds[i]
    return matches

def visualize(image, boxes=None, keypoints=None):
    fig, ax = plt.subplots(1)
    image = image.permute(1,2,0).detach().cpu().numpy()
    ax.imshow(image)
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
