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
import pandas as pd
import numpy as np
import torch
from base_utils.data_utils import get_loader
from training.inferers.utils import sliding_window_inference
from base_utils.utils import print_to_log_file
from training.coefficient import dice, dice_coef, iou_score, sensitivity, ppv
from mindspore.nn.metrics import HausdorffDistance
from datetime import datetime
import time
from monai.metrics import DiceMetric
from monai.networks import one_hot

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/data5/combineDataSets/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_resample_test.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model.pt", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
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
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
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
parser.add_argument("--logdir", default="/data3/predict_Unetr/runs/", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--logfile", default=None, type=str, help="name of log")
parser.add_argument("--organs_file", default="./networks/organs13BTCV.key", type=str, help="path of organs16AMOS.key")
parser.add_argument("--network_name", default="unetr", type=str, help="model name")
parser.add_argument("--num_samples", default=4, type=int, help="number of samples")
parser.add_argument("--mdir", default="/data3/Unetr/run/", type=str, help="directory to save the model")
parser.add_argument("--kernel_num", default=4, type=int, help="kernel_num of dynamic convolution")
parser.add_argument("--reduction", default=0.25, type=int, help="reduction of channl")
parser.add_argument("--dy_flg", action="store_true", help="use residual blocks")


def main():
    args = parser.parse_args()
    pretrained_dir = args.mdir + args.pretrained_dir
    args.test_mode = True
    organKeys = open(args.organs_file, "r").readlines()
    if args.logfile == None:
        timestamp = datetime.now()
        args.logfile = args.pretrained_dir+"/log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second)
    args.logfile = os.path.join(args.logdir,args.logfile)
    args.niidir = os.path.join(args.logdir,args.pretrained_dir)
    val_loader = get_loader(args)
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    print_to_log_file(args.logfile,args)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        if args.network_name == "unetr":
            from networks.unetr import UNETR
            model = UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=args.res_block,
                dropout_rate=args.dropout_rate,
            )
        elif args.network_name == "SEDyUnetr":
            from networks.SEDyUnetr import SEDyUnetr
            model = SEDyUnetr(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=args.res_block,
                dropout_rate=args.dropout_rate,
                reduction=0.25,
                dyAttBlocks=torch.tensor([[1.0, 1.0, 1.0, 1.0]]).cuda(),
                dy_flg=True,
                kernel_num=args.kernel_num
            )
        model_dict = torch.load(pretrained_pth)
        model.load_state_dict(model_dict["state_dict"])
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        HD_list_case = []
        IOU_list_case = []
        precision_list_case = []
        recall_list_case = []
        mean_list_case = []
        csv_data_all = []
        csv_data_name = []
        Hausdorff_metric = HausdorffDistance(percentile=95.0)
        acc_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
        res = []
        shapes = []
        for j, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print_to_log_file(args.logfile, "Inference on case {}".format(img_name))

            start = time.time()
            #inference
            val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.infer_overlap)
            end = time.time()
            res.append(end-start)
            shapes.append((val_outputs.shape[2],val_outputs.shape[3],val_outputs.shape[4]))
            print(val_outputs.shape)
            print(val_outputs.shape[2])
            val_outputs_onehot = (val_outputs == val_outputs.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
            val_outputs_onehot = val_outputs_onehot.cpu()
            val_labels_onehot = one_hot(val_labels, num_classes=16, dim=1).cpu()
            organ_Dice_list = acc_func(val_outputs_onehot, val_labels_onehot[0:,0:args.out_channels,:])
            organ_Dice_list = organ_Dice_list.cpu().numpy()[0]
            avg_acc_acc_org = np.mean([np.nanmean(l) for l in organ_Dice_list])
            print(organ_Dice_list)
            print("avg_acc_acc_org", avg_acc_acc_org)
            dice_list_sub = []
            HD_list_sub = []
            IOU_list_sub = []
            precision_list_sub = []
            recall_list_sub = []
            csv_head = []
            csv_data = []
            csv_data_name.append(img_name)
            for i in range(1, args.out_channels):
                val_o = val_outputs_onehot[0, i, :].cpu().numpy()
                val_l = val_labels_onehot[0, i, :].cpu().numpy()
                organ_Dice = organ_Dice_list[i-1]
                uni = np.unique(val_l)
                if len(uni) >1:
                    Hausdorff_metric.clear()
                    Hausdorff_metric.update(val_o, val_l, 1)
                    organ_HD = Hausdorff_metric.eval()
                    organ_IOU = iou_score(val_o, val_l)
                    organ_precision = ppv(val_o, val_l)
                    organ_recall = sensitivity(val_o, val_l)

                    dice_list_sub.append(organ_Dice)
                    HD_list_sub.append(organ_HD)
                    IOU_list_sub.append(organ_IOU)
                    precision_list_sub.append(organ_precision)
                    recall_list_sub.append(organ_recall)
                else:
                    organ_Dice = np.nan
                    organ_HD = np.nan
                    organ_IOU = np.nan
                    organ_precision = np.nan
                    organ_recall = np.nan
                    dice_list_sub.append(np.nan)
                    HD_list_sub.append(np.nan)
                    IOU_list_sub.append(np.nan)
                    precision_list_sub.append(np.nan)
                    recall_list_sub.append(np.nan)

                print_to_log_file(args.logfile, " Organ {} Dice: {}".format(i, organ_Dice))
                print_to_log_file(args.logfile, " Organ {} HD: {}".format(i, organ_HD))
                print_to_log_file(args.logfile, " Organ {} IOU: {}".format(i, organ_IOU))
                print_to_log_file(args.logfile, " Organ {} precision: {}".format(i, organ_precision))
                print_to_log_file(args.logfile, " Organ {} recall: {}".format(i, organ_recall))

                # csv
                csv_head.append(str(i)+"_Dice_organ")
                csv_head.append(str(i)+"_HD_organ")
                csv_head.append(str(i)+"_IOU_organ")
                csv_head.append(str(i)+"_precision_organ")
                csv_head.append(str(i)+"_recall_organ")
                csv_data.append(organ_Dice)
                csv_data.append(organ_HD)
                csv_data.append(organ_IOU)
                csv_data.append(organ_precision)
                csv_data.append(organ_recall)

            dice_list_sub = np.array(dice_list_sub)
            HD_list_sub = np.array(HD_list_sub)
            IOU_list_sub = np.array(IOU_list_sub)
            precision_list_sub = np.array(precision_list_sub)
            recall_list_sub = np.array(recall_list_sub)

            mean_dice = np.nanmean(dice_list_sub)
            mean_HD = np.nanmean(HD_list_sub)
            mean_IOU = np.nanmean(IOU_list_sub)
            mean_precision = np.nanmean(precision_list_sub)
            mean_recall = np.nanmean(recall_list_sub)

            mean_list_case.append([mean_dice,mean_HD, mean_IOU, mean_precision, mean_recall])
            # csv
            csv_head.append("Dice_mean")
            csv_head.append("HD_mean")
            csv_head.append("IOU_mean")
            csv_head.append("precision_mean")
            csv_head.append("recall_mean")
            csv_data.append(mean_dice)
            csv_data.append(mean_HD)
            csv_data.append(mean_IOU)
            csv_data.append(mean_precision)
            csv_data.append(mean_recall)
            csv_data_all.append(csv_data)

            print_to_log_file(args.logfile, "Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(dice_list_sub)
            print_to_log_file(args.logfile, "Mean Organ mean_HD: {}".format(mean_HD))
            HD_list_case.append(HD_list_sub)
            print_to_log_file(args.logfile, "Mean Organ mean_IOU: {}".format(mean_IOU))
            IOU_list_case.append(IOU_list_sub)
            print_to_log_file(args.logfile, "Mean Organ mean_precision: {}".format(mean_precision))
            precision_list_case.append(precision_list_sub)
            print_to_log_file(args.logfile, "Mean Organ mean_recall: {}".format(mean_recall))
            recall_list_case.append(recall_list_sub)

        dice_list_case = np.array(dice_list_case)
        HD_list_case = np.array(HD_list_case)
        IOU_list_case = np.array(IOU_list_case)
        precision_list_case = np.array(precision_list_case)
        recall_list_case = np.array(recall_list_case)
        mean_list_case = np.array(mean_list_case)

        time_sum = 0
        csv_data_name.append("Mean")
        csv_data = []
        for i in res:
            time_sum += i
        print("cont: %f"%(len(res)))
        print("FPS: %f"%(1.0/(time_sum/len(res))))

        sx = 0
        sy = 0
        sz = 0
        for i in shapes:
            sx += i[0]
            sy += i[1]
            sz += i[2]
        print(sx)
        print(sy)
        print(sz)
        print(len(shapes))
        print("avg shape: %f %f %f" % (sx / len(shapes), sy / len(shapes), sz / len(shapes)))

        for i in range(args.out_channels - 1):
            csv_data.append(np.nanmean(dice_list_case[ :,i]))
            csv_data.append(np.nanmean(HD_list_case[ :,i]))
            csv_data.append(np.nanmean(IOU_list_case[ :,i]))
            csv_data.append(np.nanmean(precision_list_case[ :,i]))
            csv_data.append(np.nanmean(recall_list_case[ :,i]))
        csv_data.append(mean_list_case[:, 0].mean())
        csv_data.append(mean_list_case[:, 1].mean())
        csv_data.append(mean_list_case[:, 2].mean())
        csv_data.append(mean_list_case[:, 3].mean())
        csv_data.append(mean_list_case[:, 4].mean())
        csv_data_all.append(csv_data)

        print_to_log_file(args.logfile, "Overall Mean Dice: {}".format(mean_list_case[:, 0].mean()))
        print_to_log_file(args.logfile, "Overall Mean mean_HD: {}".format(mean_list_case[:, 1].mean()))
        print_to_log_file(args.logfile, "Overall Mean mean_IOU: {}".format(mean_list_case[:, 2].mean()))
        print_to_log_file(args.logfile, "Overall Mean mean_precision: {}".format(mean_list_case[:, 3].mean()))
        print_to_log_file(args.logfile, "Overall Mean mean_recall: {}".format(mean_list_case[:, 4].mean()))

        csv_data_all = np.array(csv_data_all)
        csv_dict = {}
        csv_dict["sampleName"] = csv_data_name
        for j in range(len(csv_head)):
            csv_dict[csv_head[j]] = csv_data_all[:,j]
        df = pd.DataFrame(csv_dict)
        # 保存 dataframe
        df.to_csv(args.logfile+"-1.csv")

        #csv for report
        csv_dict = {}
        coefficient = ["dice","HD95","IOU","precision","recall"]
        organKeys = np.array(organKeys[0:args.out_channels-1])#BTCV 13 organs
        organKeys = np.append(organKeys, "Mean")

        col1=[]
        col2=[]
        col3=[]
        col4=[]
        col5=[]
        col6=[]
        col7=[]#dataset
        for m in range(len(csv_data_name)): #1mean + length of samples
            for j in range(args.out_channels): #1mean + length of organs
                for i in range(5): # five coefficient
                    col1.append(args.pretrained_dir) #method
                    col2.append(csv_data_name[m])
                    col3.append(coefficient[i])
                    col4.append(str(j+1))
                    col5.append(organKeys[j].replace("\n",""))
                    col6.append(csv_data_all[m][i+5*j])
                    if csv_data_name[m][0:4]=="ABD_":
                        col7.append("1")
                    elif csv_data_name[m][0:8]=="Patient_":
                        col7.append("2")
                    elif csv_data_name[m][0:7]=="spleen_":
                        col7.append("3")
                    elif csv_data_name[m][0:4]=="Mean":
                        col7.append("Mean")
                    else:
                        col7.append("others")
        csv_dict["method"] = col1
        csv_dict["dataset"] = col7
        csv_dict["sampleName"] = col2
        csv_dict["coefficient"] = col3
        csv_dict["organ"] = col4
        csv_dict["organName"] = col5
        csv_dict["val"] = col6
        df = pd.DataFrame(csv_dict)
        # 保存 dataframe
        df.to_csv(args.logfile+"-report.csv")

if __name__ == "__main__":
    main()
