from multiprocessing import Pool
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from coperception.utils.postprocess import *


def average_precision(recalls, precisions, mode="area"):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == "11points":
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.
    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.
    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann["labels"] == class_id

        if ann["bboxes"].ndim == 1:
            cls_gts.append(ann["bboxes"][0])
        else:
            cls_gts.append(ann["bboxes"][gt_inds, :])

        if ann.get("labels_ignore", None) is not None:
            ignore_inds = ann["labels_ignore"] == class_id

            if ann["bboxes_ignore"].ndim == 1:
                cls_gts_ignore.append(ann["bboxes_ignore"][0])
            else:
                cls_gts_ignore.append(ann["bboxes_ignore"][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 8), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore


def print_map_summary(mean_ap, results, dataset=None, scale_ranges=None, logger=None):
    """Print mAP and results of each class.
    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.
    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
    """

    if logger == "silent":
        return

    if isinstance(results[0]["ap"], np.ndarray):
        num_scales = len(results[0]["ap"])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result["recall"].size > 0:
            recalls[:, i] = np.array(cls_result["recall"], ndmin=2)[:, -1]
        aps[:, i] = cls_result["ap"]
        num_gts[:, i] = cls_result["num_gts"]

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ["class", "gts", "dets", "recall", "ap"]
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f"Scale range {scale_ranges[i]}", logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j],
                num_gts[i, j],
                results[j]["num_dets"],
                f"{recalls[i, j]:.3f}",
                f"{aps[i, j]:.3f}",
            ]
            table_data.append(row_data)
        table_data.append(["mAP", "", "", "", f"{mean_ap[i]:.3f}"])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log("\n" + table.table, logger=logger)
