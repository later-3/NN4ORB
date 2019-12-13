import argparse
import os
import numpy as np
import math
import itertools
import sys
import tensorwatch as tw
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from utils import evaluate
from tensorboardX  import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from orb_model import *
from data_hanlder import *
from torchviz import make_dot
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
parser.add_argument("--dataset_path", type=str, default='', help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=24, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--height", type=int, default=128, help="high res. image height")
parser.add_argument("--width", type=int, default=128, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--evaluation_interval", type=int, default=5, help="interval between model evaluation")
opt = parser.parse_args()
print(opt)
os.makedirs("checkpoints", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training = False
cuda = torch.cuda.is_available()

shape = (opt.height, opt.width)

# Initialize net
orbresnet = orbResNet()

# Losses

if cuda:
    orbresnet = orbresnet.cuda()

if opt.epoch != 0:
   orbresnet.load_state_dict(torch.load(f"checkpoints/orbresnet_ckpt_%d.pth"%(opt.epoch-1)))


# Optimizers
optimizer = torch.optim.Adam(orbresnet.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2))
# optimizer = torch.optim.Adam(orbresnet.parameters())

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataset = DataProducer(opt.dataset_path, img_shape=shape)
dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    collate_fn=dataset.collate_fn,
)

metrics = [
    "loss",
    "loss_angle",
    "loss_response",
    "precision",
    "loss_conf",
    "conf_obj",
    "conf_noobj"
]
logger = Logger("logs")


# ----------
#  Training
# ----------
if training:
    with  SummaryWriter(comment='loss') as writer:
        for epoch in range(opt.epoch, opt.n_epochs):
            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                batches_done = len(dataloader) * epoch + batch_i
                # Configure model input
                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device))

                # ------------------
                #  Train
                # ------------------

                optimizer.zero_grad()

                # Generate a high resolution image from low resolution input
                loss, outputs = orbresnet(imgs, targets)

                writer.add_scalar('runs/loss', loss, epoch)

                loss.backward()
                if batches_done % opt.gradient_accumulations:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                # --------------
                #  Log Progress
                # --------------
                #log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [-conf_obj: %f] [conf_noobj: %f] "
                    "[response: %f] [conf: %f] [precision: %f]  [angle: %f]\n"
                    % (epoch, opt.n_epochs, batch_i, len(dataloader), loss.item(),
                       orbresnet.losslayer.metrics["conf_obj"], orbresnet.losslayer.metrics["conf_noobj"], orbresnet.losslayer.metrics["loss_response"],
                    orbresnet.losslayer.metrics["loss_conf"], orbresnet.losslayer.metrics["precision"], orbresnet.losslayer.metrics["loss_angle"])
                )

            if epoch % opt.checkpoint_interval == 0:
                torch.save(orbresnet.state_dict(), f"checkpoints/log_orbresnet_ckpt_%d_loss%f.pth" % (epoch, loss.item()))
