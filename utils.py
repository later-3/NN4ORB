from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data_hanlder import *

def to_cpu(tensor):
    return tensor.detach().cpu()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def build_targets(batch_size, map_size, target, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if target.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if target.is_cuda else torch.FloatTensor

    nB = batch_size
    nG = map_size

    # Output tensors
    obj_mask = ByteTensor(nB, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nG, nG).fill_(1)

    tx = FloatTensor(nB, nG, nG).fill_(0)
    ty = FloatTensor(nB, nG, nG).fill_(0)
    tangle = FloatTensor(nB, nG, nG).fill_(0)
    tresponse = FloatTensor(nB, nG, nG).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 1:]
    gxy = target_boxes[:, :2]
    gangle = target_boxes[:, 2].t()
    gresponse = target_boxes[:, 3].t()
    b = target[:, 0].long().t()
    gx, gy = gxy.t()
    gi, gj = (gxy * torch.tensor(128, dtype=torch.long)).long().t()
    # Set masks
    obj_mask[b, gj, gi] = 1
    noobj_mask[b, gj, gi] = 0
    # Coordinates
    tx[b,gj, gi] = gx
    ty[b, gj, gi] = gy

    # angle  response
    tangle[b, gj, gi] = gangle
    tresponse[b, gj, gi] = gresponse

    tconf = obj_mask.float()
    return obj_mask, noobj_mask, tx, ty, tangle, tresponse, tconf

def evaluate(model, path, conf_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = DataProducer(path, img_shape=img_size, augment=False, dataset='validation_set.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="predict features")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst