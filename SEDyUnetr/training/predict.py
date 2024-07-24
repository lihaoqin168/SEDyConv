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

import argparse
import os

import numpy as np
import torch
from training.trainer import dice
from base_utils.data_utils import get_loader
from base_utils.nii_utils import saveToNii
from base_utils.utils import print_to_log_file
from training.inferers.utils import sliding_window_inference

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="../pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/data3/combineDataSets/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model.pt", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.7, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--organs_file", default="../networks/organs.key", type=str, help="path of organs.key")
parser.add_argument("--dyh_channel", default=16, type=int, help="dyh_channel dimention")
parser.add_argument("--logdir", default="/data3/DyUnetr/inference_log", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--logfile", default=None, type=str, help="name of log")


def main():
    args = parser.parse_args()
    args.pretrained_dir = "../runs/dyunetr_20220730_roi128/"
    args.pretrained_model_name = "model.pt"
    args.test_mode = True
    if args.logfile == None:
        from datetime import datetime
        timestamp = datetime.now()
        args.logfile = args.pretrained_dir+args.pretrained_model_name+"_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second)
    args.logfile = os.path.join(args.logdir,args.logfile)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    out_channels = len(open(args.organs_file,"r").readlines())+1
    # init organsKeys=[0,1,2,3,4,5,6,7,8,9...]
    organsKeys = []
    organsKeys = [[0]]
    # organsKeys.append([i for i in range(20)])
    # organsKeys.append([i for i in range(16)])
    # organsKeys.append([i for i in range(14)])
    # organsKeys = [[0, 5, 8, 14, 15]]
    print_to_log_file(args.logfile,args)
    print_to_log_file(args.logfile, "organsKeys", organsKeys)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":

        # from networks.dyunetr_taskencoding2 import DyUNETR
        # from networks.dyunetr import DyUNETR
        from networks.dyunetr_roi128 import DyUNETR
        model = DyUNETR(
            in_channels=args.in_channels,
            out_channels=out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            dyh_channel=args.dyh_channel,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
        weight = torch.load(pretrained_pth)
        if ("state_dict" in [entry for entry in weight.keys()]):
            weight = weight["state_dict"]
        # model.load_from(weights=weight)
        model.load_state_dict(weight)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print_to_log_file(args.logfile, "Inference on case {}".format(img_name))
            # save Nii
            saveToNii("data_{}".format(img_name), val_inputs[0, :].cpu(), out_dir="/data3/DyUnetr/inference/")
            saveToNii("target_{}".format(img_name), val_labels[0, :].cpu(), sitkcast=True, out_dir="/data3/DyUnetr/inference/")
            #inference
            val_outputs = sliding_window_inference((val_inputs,organsKeys), (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.infer_overlap)

            print("val_inputs.shape", val_inputs.shape)#torch.Size([1, 1, 238, 190, 246])
            print("val_outputs.shape", val_outputs.shape)#torch.Size([1, 1, 238, 190, 246])

            #save Nii
            saveToNii("predict_{}".format(img_name), val_outputs[0, :].cpu(), sitkcast=True, out_dir="/data3/DyUnetr/inference/")
            # calculate dice
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            for i in range(1, out_channels):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                print_to_log_file(args.logfile, " Organ {} Dice: {}".format(i, organ_Dice))
                if organ_Dice > 0:
                    dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print_to_log_file(args.logfile, "Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)



        print_to_log_file(args.logfile, "Overall Mean Dice: {}".format(np.mean(dice_list_case)))


if __name__ == "__main__":
    main()
