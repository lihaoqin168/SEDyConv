# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from time import time, sleep
from datetime import datetime
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

base = os.environ['nnUNet_raw_data_base'] if "nnUNet_raw_data_base" in os.environ.keys() else None
try:
    nnUNet_raw_data = os.path.join(base, "nnUNet_raw_data")
except:
    print("you need set environment. for example: export nnUNet_raw_data_base=/data/rawdata")

# def resample_3d(img, target_size):
#     imx, imy, imz = img.shape
#     tx, ty, tz = target_size
#     zoom_ratio = ( float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
#     img_resampled = ndimage.zoom( img, zoom_ratio, order=0, prefilter=False)
#     return img_resampled

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def print_to_log_file(log_file, *args, also_print_to_console=True, add_timestamp=True):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)
    output_folder = os.path.dirname(os.path.realpath(log_file))
    if add_timestamp:
        args = ("%s:" % dt_object, *args)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(log_file):
        # timestamp = datetime.now()
        # log_file = os.path.join(output_folder, "log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
        #                      (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
        #                       timestamp.second))
        with open(log_file, 'w') as f:
            f.write("Starting... \n")
    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(log_file, 'a+') as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            import sys
            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*args)



def get_temperature(iteration, epoch, iter_per_epoch, temp_epoch=10, temp_init=30.0):
    total_temp_iter = iter_per_epoch * temp_epoch
    current_iter = iteration + epoch * iter_per_epoch
    temperature = 1.0 + max(0, (temp_init - 1.0) * ((total_temp_iter - current_iter) / total_temp_iter))
    return temperature