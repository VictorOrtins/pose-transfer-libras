# imports
import datetime
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import create_dataloaderV2
from models.pose_transfer_model import PoseTransferModel
import os
import sys
import cv2

import random
import numpy as np
import torch

from collections import OrderedDict

def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(0)


def add_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict['module.' + k] = v
    return new_state_dict

# configurations
# -----------------------------------------------------------------------------
dataset_name = 'Libras'

dataset_root = sys.argv[1]
csv_path = sys.argv[2]
rgb_frames_path = sys.argv[3]
json_keypoints_path = sys.argv[4]

img_pairs_train = f'{dataset_root}/{csv_path}'
# img_pairs_test = f'{dataset_root}/test_img_pairs.csv'

# pose_maps_dir_train = f'{dataset_root}/train_pose_maps'
# pose_maps_dir_test = f'{dataset_root}/test_pose_maps'

rgb_frames_dir_train = f'{dataset_root}/{rgb_frames_path}'
# json_keypoints_dir_train = f'{dataset_root}/openpose_output_train/json'
json_keypoints_dir_train = f'{dataset_root}/{json_keypoints_path}'

# rgb_frames_dir_teste = f'{dataset_root}/frames_videos_teste'
# json_keypoints_dir_teste = f'{dataset_root}/openpose_output_teste/json'

gpu_ids = [0]
# gpu_ids = []

batch_size_train = 8
batch_size_test = 8
n_epoch = 100
out_freq = 500

ckpt_id = None
ckpt_dir = None

run_info = ''
out_path = sys.argv[5]
os.makedirs(out_path, exist_ok=True)
# -----------------------------------------------------------------------------


# create timestamp and infostamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
infostamp = f'_{run_info.strip()}' if run_info.strip() else ''

# create tensorboard logger
logger = SummaryWriter(f'{out_path}/runs/{timestamp}{infostamp}')

# create transforms
img_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
map_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# create dataloaders
train_dataloader = create_dataloaderV2(rgb_frames_dir_train, json_keypoints_dir_train, img_pairs_train,
                                     img_transform, map_transform,
                                     batch_size=batch_size_train, shuffle=True)

# test_dataloader = create_dataloaderV2(rgb_frames_dir_teste, json_keypoints_dir_teste, img_pairs_test,
#                                     img_transform, map_transform,
#                                     batch_size=batch_size_test, shuffle=False)

# create fixed batch for testing
# fixed_test_batch = next(iter(test_dataloader))

model = PoseTransferModel(gpuids=gpu_ids, keypoints_numbers=274)

model.print_networks(verbose=False)

# load pretrained weights into model
#if ckpt_id and ckpt_dir:
#    model.load_networks(ckpt_dir, ckpt_id, verbose=True)

# train model
n_batch = len(train_dataloader)
w_batch = len(str(n_batch))
w_epoch = len(str(n_epoch))
n_iters = 0

os.makedirs(f'{out_path}/test_samples/{timestamp}{infostamp}', exist_ok=True)
out_freq_change_weight = 6000

for epoch in range(n_epoch):
    for batch, data in enumerate(train_dataloader):

        if ((n_iters + 1) % out_freq_change_weight == 0):
            train_dataloader.dataset.weight_another = 0.6

        if ((n_iters + 1) % (out_freq_change_weight + 6000) == 0):
            train_dataloader.dataset.weight_pose = 1.5
            train_dataloader.dataset.weight_another = 1.0

        time_0 = time.time()
        model.set_inputs(data)
        model.optimize_parameters()
        losses = model.get_losses()
        loss_G = losses['lossG']
        lossG_L1 = losses["lossG_L1"]
        lossG_GAN = losses["lossG_GAN"]
        lossG_PER = losses["lossG_PER"]
        loss_D = losses['lossD']
        time_1 = time.time()
        print(f'[TRAIN] Epoch:{epoch+1:{w_epoch}d}/{n_epoch} | Batch:{batch+1:{w_batch}d}/{n_batch} |',
              f'LossG:{loss_G:7.4f} | LossG_L:{lossG_L1:7.4f} | LossG_G:{lossG_GAN:7.4f} | LossG_P:{lossG_PER:7.4f} | LossD: {loss_D:7.4f} | Time: {round(time_1-time_0, 2):.2f} sec |')
        
        if (n_iters % out_freq == 0) or (batch+1 == n_batch and epoch+1 == n_epoch):
            model.save_networks(f'{out_path}/ckpt/{timestamp}{infostamp}', n_iters, verbose=True)
            for loss_name, loss in losses.items():
                loss_group = 'LossG' if loss_name.startswith('lossG') else 'LossD'
                logger.add_scalar(f'{loss_group}/{loss_name}', loss, n_iters)
            
            # model.set_inputs(fixed_test_batch)
            # visuals = model.compute_visuals()
            # cv2.imwrite(f"{out_path}/test_samples/{timestamp}{infostamp}/Iteration_{n_iters}.jpg", visuals)
            #logger.add_image(f'Iteration_{n_iters}', visuals, n_iters)
        
        n_iters += 1
