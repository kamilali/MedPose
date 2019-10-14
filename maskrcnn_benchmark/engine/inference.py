# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList

import copy
import shutil
import json

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
            for im_id, im_prediction in zip(original_id, prediction):
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
                keypoint_scores = keypoints.get_field('logits')
                keypoints = keypoints.resize((vid_width, vid_height))
                #print(keypoints.keypoints.tolist())
                scores = im_prediction.get_field('scores').tolist()
                labels = im_prediction.get_field('labels').tolist()

                # match per region and create annotation
                #print(len(keypoints.keypoints))
                ann_key = str(im_id)
                # check if annotation exists for current image
                if ann_key in annotation_mappings:
                    a_idxs = annotation_mappings[ann_key]
                    track_id = 0
                    for a_idx in a_idxs:
                        gt_ann = annotation_json[a_idx]
                        # print(gt_ann.keys(), "ground truth annotation keys")
                        gt_box = BoxList([gt_ann['bbox']], (vid_width, vid_height), mode='xywh')
                        box_ious = boxlist_iou(im_prediction, gt_box)
                        # ensure that a corresponding box exists
                        if box_ious.shape[0] != 0:
                            match_idx = box_ious.max(0)[1].item()
                            #print(gt_ann['bbox'], "ground truth box")
                            #print(boxes[match_idx], "predicted box")
                            pred_ann = {}
                            pred_ann['bbox'] = boxes[match_idx]
                            pred_ann['keypoints'] = keypoints.keypoints[match_idx].numpy().flatten().tolist()
                            #pred_ann['id'] = gt_ann['id']
                            #print(gt_ann['id'], "annotation id")
                            #print(im_id, "image id")
                            # currently only estimation not tracking
                            pred_ann['track_id'] = track_id
                            track_id = track_id + 1
                            #pred_ann['track_id'] = gt_ann['track_id']
                            pred_ann['image_id'] = im_id
                            #pred_ann['bbox_head'] = gt_ann['bbox_head']
                            #pred_ann['category_id'] = gt_ann['category_id']
                            #pred_ann['scores'] = [scores[match_idx]]
                            pred_ann['scores'] = keypoint_scores[match_idx].numpy().flatten().tolist()
                            # print(pred_ann.keys(), "predicted annotation keys")
                            #print(a_idx)
                            #predicted_json["annotations"].append(pred_ann)
                            predicted_json["annotations"][a_idx] = pred_ann
                            #print(gt_ann['scores'], gt_ann['category_id'], gt_ann['id'], gt_ann['image_id'], gt_ann['track_id'], gt_ann['bbox_head'])
                            #pred_ann['scores']
                            #print(len(gt_ann['keypoints']), "ground truth keypoints")
                            #print(len(pred_ann['keypoints']), "predicted keypoints")
                else:
                    print("annotation does not exist for current image... no predictions")
            # write out prediction file for current video 
            output_file = os.path.join(output_folder, dataset.id_to_annotation_file_map[vid_id])
            print("Saving current video predictions to", output_file)
            with open(output_file, 'w') as f:
                json.dump(predicted_json, f)

        # pose track evaluation handled seperately
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
