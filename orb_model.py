import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

from utils import build_targets, to_cpu, non_max_suppression

class LossLayer(nn.Module):

    # TODO rename num_classes
    def __init__(self, anchors=None, num_classes=3, img_dim=128):
        super(LossLayer, self).__init__()
        self.anchors = anchors
        # self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 30
        self.response_scale = 1
        self.angle_scale = 1
        self.noobj_scale = 1000
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size


    def forward(self, x, targets=None, img_dim=None):

        # Conf angle response
        # x y angle response

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        pre_dim = x.size(1)
        map_size = x.size(2)

        prediction = (
            x.view(num_samples, pre_dim, map_size, map_size)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        # Get outputs
        pred_conf = torch.sigmoid(prediction[..., 0])  # Conf
        pred_angle = torch.sigmoid(prediction[..., 1])  # angle

        pred_response = torch.sigmoid(prediction[..., 2])  # response

        pred_boxes = FloatTensor(prediction[..., :3].shape)
        pred_boxes[..., 0] = pred_conf.data
        pred_boxes[..., 1] = pred_angle.data
        pred_boxes[..., 2] = pred_response.data

        output = pred_boxes

        if targets is None:
            return 0, output
        else:
            obj_mask, noobj_mask, tx, ty, tangle, tresponse, tconf = build_targets(
                batch_size=num_samples,
                map_size=map_size,
                target=targets,
                ignore_thres=self.ignore_thres,
            )

            obj_mask = obj_mask.bool()
            noobj_mask = noobj_mask.bool()

            loss_angle = self.mse_loss(pred_angle[obj_mask], tangle[obj_mask])
            loss_response = self.mse_loss(pred_response[obj_mask], tresponse[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            ll = -100 * torch.log(pred_conf[obj_mask].mean())
            l2 = 100 * torch.log(pred_conf[noobj_mask].mean())
            total_loss =  loss_angle +  loss_response +  loss_conf  + ll

            # Metrics
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = ( (pred_conf[obj_mask] > 0.5)).sum()

            len_tfeature = targets.size(0)

            precision =  to_cpu(conf50).item() / len_tfeature # / tconf.size()

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "loss_angle": to_cpu(loss_angle).item(),
                "loss_response": to_cpu(loss_response).item(),
                "loss_conf": to_cpu(loss_conf).item(),
                "precision": precision,
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
            }

            return total_loss, output

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features,  momentum=0.8),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features,  momentum=0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class orbResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(orbResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, momentum=0.8))

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.losslayer = LossLayer()

    def forward(self, x, targets=None):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.conv3(out)
        loss, output = self.losslayer(out, targets)

        return loss, output




