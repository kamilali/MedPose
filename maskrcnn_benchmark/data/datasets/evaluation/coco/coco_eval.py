import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    posetrack_eval
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit, posetrack_eval=posetrack_eval
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset, posetrack_eval)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset, posetrack_eval)
    if 'keypoints' in iou_types:
        logger.info('Preparing keypoints results')
        coco_results['keypoints'] = prepare_for_coco_keypoint(predictions, dataset, posetrack_eval)

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            if not posetrack_eval:
                res = evaluate_predictions_on_coco(
                    dataset.coco, coco_results[iou_type], file_path, iou_type
                )
                results.update(res)
            else:
                file_paths = []
                if output_folder:
                    for idx in range(len(dataset.coco_objs)):
                        file_path = os.path.join(output_folder, iou_type + "_" + str(idx) + ".json")
                        file_paths.append(file_path)
                res = evaluate_predictions_on_posetrack(
                    dataset.coco_objs, coco_results[iou_type], file_paths, dataset.del_idxs, iou_type
                )
                results.update(res)
    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results


def prepare_for_coco_detection(predictions, dataset, posetrack_eval):
    # assert isinstance(dataset, COCODataset)
    full_coco_results = []
    for image_id, prediction in enumerate(predictions):
        if not posetrack_eval:
            original_id = [dataset.id_to_img_map[image_id]]
            prediction = [prediction]
        else:
            original_id = dataset.id_to_video_map[image_id]
            #original_coco = dataset.coco_objs[image_id]
        coco_results = []
        for im_id, im_prediction in zip(original_id, prediction):
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            im_prediction = im_prediction.resize((image_width, image_height))
            im_prediction = im_prediction.convert("xywh")

            boxes = im_prediction.bbox.tolist()
            scores = im_prediction.get_field("scores").tolist()
            labels = im_prediction.get_field("labels").tolist()

            mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

            if posetrack_eval:
                #curr_frame = original_coco.loadImgs(im_id)[0]
                #curr_frame = Image.open(os.path.join('datasets/posetrack', curr_frame['file_name'])).convert("RGB")
                #visualize(curr_frame, boxes, None)
                #exit()
                coco_results.extend(
                    [
                        {
                            "image_id": im_id,
                            "category_id": mapped_labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            else:
                full_coco_results.extend(
                    [
                        {
                            "image_id": im_id,
                            "category_id": mapped_labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
        if posetrack_eval:
            full_coco_results.append(coco_results)

    return full_coco_results


def prepare_for_coco_segmentation(predictions, dataset, posetrack_eval):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def prepare_for_coco_keypoint(predictions, dataset, posetrack_eval):
    # assert isinstance(dataset, COCODataset)
    full_coco_results = []
    for image_id, prediction in enumerate(predictions):
        if not posetrack_eval:
            original_id = [dataset.id_to_img_map[image_id]]
            prediction = [prediction]
        else:
            original_id = dataset.id_to_video_map[image_id]
            #original_coco = dataset.coco_objs[image_id]
 
        # TODO replace with get_img_info?
        coco_results = []
        for im_id, im_prediction in zip(original_id, prediction):
            if len(im_prediction.bbox) == 0:
                continue
            img_info = dataset.get_img_info(image_id)
            image_width = img_info['width']
            image_height = img_info['height']
            im_prediction = im_prediction.resize((image_width, image_height))
            im_prediction = im_prediction.convert('xywh')

            boxes = im_prediction.bbox.tolist()
            scores = im_prediction.get_field('scores').tolist()
            labels = im_prediction.get_field('labels').tolist()
            keypoints = im_prediction.get_field('keypoints')
            keypoints = keypoints.resize((image_width, image_height))
            keypoints = keypoints.keypoints.view(keypoints.keypoints.shape[0], -1).tolist()

            mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
            if posetrack_eval:
                # curr_frame = original_coco.loadImgs(im_id)[0]
                # curr_frame = Image.open(os.path.join('datasets/posetrack', curr_frame['file_name'])).convert("RGB")
                # visualize(curr_frame, None, keypoints)
                # exit()
                coco_results.extend([{
                    'image_id': im_id,
                    'category_id': mapped_labels[k],
                    'keypoints': keypoint,
                    'score': scores[k]} for k, keypoint in enumerate(keypoints)])
            else:
                full_coco_results.extend([{
                    'image_id': im_id,
                    'category_id': mapped_labels[k],
                    'keypoints': keypoint,
                    'score': scores[k]} for k, keypoint in enumerate(keypoints)])

        if posetrack_eval:
            full_coco_results.append(coco_results)

    return full_coco_results

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None, posetrack_eval=False
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        if not posetrack_eval:
            original_id = dataset.id_to_img_map[image_id]
        else:
            original_id = dataset.id_to_video_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_posetrack(
    coco_gts, coco_results, json_result_files, del_idxs, iou_type="bbox"
):
    import numpy as np
    import json

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    print(coco_gts[0].anns)
    exit()

    coco_evals = []
    total_coco_gt, total_coco_dt = None, None
    coco_gts = [coco_gts[i] for i in range(len(coco_gts)) if i not in del_idxs]
    json_result_files = [json_result_files[i] for i in range(len(json_result_files)) if i not in del_idxs]
    coco_results = [coco_results[i] for i in range(len(coco_results)) if i not in del_idxs]
    
    # for testing purposes only
    #coco_gts = coco_gts[:5]
    #json_result_files = json_result_files[:5]

    for json_result_file, coco_result in zip(json_result_files, coco_results):
        with open(json_result_file, "w") as f:
            json.dump(coco_result, f)
    
    for curr_idx, (coco_gt, json_result_file, coco_result) in enumerate(zip(coco_gts, json_result_files, coco_results)):
        load_allowed = True
        for idx in coco_gt.anns:
            if iou_type == "bbox":
                if 'bbox' in coco_gt.anns[idx]:
                    bb = coco_gt.anns[idx]['bbox']
                    coco_gt.anns[idx]['iscrowd'] = 0
                    coco_gt.anns[idx]['area'] = bb[2]*bb[3]
                    load_allowed = True
                else:
                    print('could not find relevant bbox annotation... skipping')
                    load_allowed = False
                    break
            if iou_type == "keypoints":
                s = coco_gt.anns[idx]['keypoints']
                x = s[0::3]
                y = s[1::3]
                v = s[2::3]
                num_keypoints = v.count(1)
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                coco_gt.anns[idx]['area'] = (x1-x0)*(y1-y0)
                coco_gt.anns[idx]['bbox'] = [x0,y0,x1-x0,y1-y0]
                coco_gt.anns[idx]['num_keypoints'] = num_keypoints
                coco_gt.anns[idx]['iscrowd'] = 0
        if load_allowed:
            coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_result else COCO()
            #coco_dt = coco_gt.loadRes(coco_results)
            # print(len(coco_dt.dataset), len(coco_dt.dataset))
            # print(coco_gt.dataset.keys(), coco_dt.dataset.keys(), "dataset")
            # print(len(coco_gt.anns), len(coco_dt.anns))
            # print(coco_gt.anns.keys(), coco_dt.anns.keys(), "annotations")
            # print(len(coco_gt.cats), len(coco_dt.cats))
            # print(coco_gt.cats.keys(), coco_dt.cats.keys(), "categories")
            # print(len(coco_gt.imgs), len(coco_dt.imgs))
            # print(coco_gt.imgs.keys(), coco_dt.imgs.keys(), "images")
            # if iou_type == 'keypoints':
            #     for gt_key, dt_key in zip(coco_gt.anns.keys(), coco_dt.anns.keys()):
            #         print(gt_key, dt_key)
            #         print(coco_gt.anns[gt_key])
            #         print(coco_dt.anns[dt_key])
            #         image_id = coco_dt.anns[dt_key]['image_id']
            #         other_image_id = coco_gt.anns[gt_key]['image_id']
            #         bbox = coco_dt.anns[dt_key]['bbox']
            #         keypoints = coco_dt.anns[dt_key]['keypoints']
            #         other_bbox = coco_gt.anns[gt_key]['bbox']
            #         other_keypoints = coco_gt.anns[gt_key]['keypoints']
            #         img = coco_gt.loadImgs(ids=image_id)[0]
            #         other_img = coco_gt.loadImgs(ids=other_image_id)[0]
            #         print(img['file_name'])
            #         img = Image.open(os.path.join('datasets/posetrack', img['file_name'])).convert("RGB")
            #         other_img = Image.open(os.path.join('datasets/posetrack', other_img['file_name'])).convert("RGB")
            #         visualize(img, [bbox], [keypoints])
            #         visualize(other_img, [other_bbox], [other_keypoints])
            if total_coco_gt is None:
                total_coco_gt = coco_gt
            else:
                total_coco_gt.dataset.update(coco_gt.dataset)
                total_coco_gt.anns.update(coco_gt.anns)
                total_coco_gt.cats.update(coco_gt.cats)
                total_coco_gt.imgs.update(coco_gt.imgs)
                total_coco_gt.imgToAnns.update(coco_gt.imgToAnns)
                for cat in coco_gt.catToImgs.keys():
                    if cat in total_coco_gt.catToImgs:
                        for gt_img in coco_gt.catToImgs[cat]:
                            total_coco_gt.catToImgs[cat].append(gt_img)
                    else:
                        total_coco_gt.catToImgs[cat] = coco_gt.catToImgs[cat]

            if total_coco_dt is None:
                total_coco_dt = coco_dt
            else:
                total_coco_dt.dataset.update(coco_dt.dataset)
                total_coco_dt.anns.update(coco_dt.anns)
                total_coco_dt.cats.update(coco_dt.cats)
                total_coco_dt.imgs.update(coco_dt.imgs)
                total_coco_dt.imgToAnns.update(coco_dt.imgToAnns)
                for cat in coco_dt.catToImgs.keys():
                    if cat in total_coco_dt.catToImgs:
                        for dt_img in coco_dt.catToImgs[cat]:
                            total_coco_dt.catToImgs[cat].append(dt_img)
                    else:
                        total_coco_dt.catToImgs[cat] = coco_dt.catToImgs[cat]
            
            #coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            #coco_eval.evaluate()
            #coco_eval.accumulate()
            #coco_eval.summarize()
            #coco_evals.append(coco_eval)
        else:
            print("skipped")
   
    coco_eval = COCOeval(total_coco_gt, total_coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    #exit()
    return coco_eval

def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)

def visualize(image, boxes=None, keypoints=None):
    fig, ax = plt.subplots(1)
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
