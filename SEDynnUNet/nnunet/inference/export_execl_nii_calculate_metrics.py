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
import pandas as pd
import numpy as np
from nnunet.base_utils.utils import print_to_log_file
from nnunet.training.coefficient import dice, dice_coef, iou_score, sensitivity, ppv
from mindspore.nn.metrics import HausdorffDistance
from monai.metrics import DiceMetric
from monai.networks import one_hot
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import torch


parser = argparse.ArgumentParser(description="segmentation calculate metrics")
parser.add_argument("--method", default="nnUnet", type=str, help="name of method")
# parser.add_argument("--data_dir", default="/data3/predict_nii_nnUnet-model_final_checkpoint/nnUNetTrainerV2_DyQK_V5_Cinout_2k2_noKernel_N1-018", type=str, help="dataset directory")
# parser.add_argument("--data_dir", default="/data3/predict_nii_nnUnet-model_best/nnUNetTrainerV2_DyQK_V5_Cinout_2k2_noKernel_strideXall_N4-018", type=str, help="dataset directory")
parser.add_argument("--data_dir", default="/data3/predict_nii_nnUnet/018/SEDynnUNetTrainerV2-018", type=str, help="dataset directory")
# parser.add_argument("--logdir", default="/data3/predict_nii_nnUnet/nnUNetTrainerV2_DyQK_V5_Cinout_2k2_NM1normal-018/", type=str, help="directory to save the tensorboard logs")
# parser.add_argument("--json_dir", default="/data4/nnUNet_raw_data_base/nnUNet_raw_data/Task055_SegTHOR/", type=str, help="dataset directory")
parser.add_argument("--json_dir", default="/data4/nnUNet_raw_data_base/nnUNet_raw_data/Task018_AbdominalOrganSegmentation/", type=str, help="dataset directory")
# parser.add_argument("--json_dir", default="/data4/nnUNet_raw_data_base/nnUNet_raw_data/Task007_Pancreas/", type=str, help="dataset directory")
# parser.add_argument("--json_dir", default="/data4/nnUNet_raw_data_base/nnUNet_raw_data/Task185_esophagus/", type=str, help="dataset directory")
# parser.add_argument("--json_dir", default="/data4/nnUNet_raw_data_base/nnUNet_raw_data/Task160_Kvasir_SEG/", type=str, help="dataset directory")
# parser.add_argument("--json_dir", default="/data4/nnUNet_raw_data_base/nnUNet_raw_data/Task136_AbdomenCT1K/", type=str, help="dataset directory")
# parser.add_argument("--json_dir", default="/data4/nnUNet_raw_data_base/nnUNet_raw_data/Task135_AMOS/", type=str, help="dataset directory")
# parser.add_argument("--json_dir", default="/data4/nnUNet_raw_data_base/nnUNet_raw_data/Task141_Totalsegmentator/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset.json", type=str, help="dataset json file")
# parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
# parser.add_argument("--out_channels", default=5, type=int, help="number of output channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
# parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
# parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
# parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--logfile", default=None, type=str, help="name of log")
parser.add_argument("--organs_file", default="organs14BTCV.key", type=str, help="path of organs.key")
# parser.add_argument("--organs_file", default="organs15AMOS.key", type=str, help="path of organs.key")
# parser.add_argument("--organs_file", default="organsTask136.key", type=str, help="path of organs.key")
# parser.add_argument("--organs_file", default="organsSegTHOR.key", type=str, help="path of organs.key")
# parser.add_argument("--organs_file", default="organs4Abdomen1K.key", type=str, help="path of organs.key")
# parser.add_argument("--organs_file", default="organs2.key", type=str, help="path of organs.key")
# parser.add_argument("--organs_file", default="organs1.key", type=str, help="path of organs.key")


def main():
    args = parser.parse_args()
    if args.logfile == None:
        from datetime import datetime
        timestamp = datetime.now()
        args.logfile = args.data_dir+"/log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second)
    args.logdir = args.data_dir
    args.logfile = os.path.join(args.logdir,args.logfile)
    args.niidir = os.path.join(args.logdir,'output_nii')
    organKeys = open(args.organs_file, "r").readlines()
    print_to_log_file(args.logfile,args)
    datalist_json = os.path.join(args.json_dir, args.json_list)
    if os.path.exists(datalist_json):
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

        with open(datalist_json, 'r') as datalist_json:
            json_data = json.load(datalist_json)
        for test_p in json_data['test']:
            label_nii_filename = join(args.json_dir,test_p.replace('imagesTs','labelsTs'))
            # label_nii_filename = join(args.json_dir,test_p.replace('imagesTs','labelsTs018'))
            # label_nii_filename = join(args.json_dir,test_p.replace('imagesTs','labelsTs135'))
            img_name = test_p.split("/")[-1]#[:-7]
            val_labels = sitk.ReadImage(label_nii_filename)
            # val_img = sitk.ReadImage(join(args.json_dir,test_p.replace('.nii.gz','_0000.nii.gz')))
            val_labels  = sitk.GetArrayFromImage(val_labels).astype(np.int32)
            val_labels = torch.from_numpy(val_labels).unsqueeze(0).unsqueeze(0)
            val_outputs = sitk.ReadImage(join(args.data_dir,img_name))
            val_outputs = torch.from_numpy(sitk.GetArrayFromImage(val_outputs)).unsqueeze(0).unsqueeze(0)

            print_to_log_file(args.logfile, "Inference on case {}".format(img_name))
            # save Nii

            # from base_utils.nii_utils import saveToNii
            # saveToNii("data_{}".format(img_name), val_inputs[0, :].cpu(), rot180=True, out_dir=args.niidir)
            # saveToNii("target_{}".format(img_name), val_labels[0, :].cpu(), rot180=True, sitkcast=True,
            #           out_dir=args.niidir)
            #onehot
            print("np.unique(val_labels)", np.unique(val_labels))
            val_labels_onehot = one_hot(val_labels, num_classes=args.out_channels, dim=1)
            val_outputs_onehot = one_hot(val_outputs, num_classes=args.out_channels, dim=1)
            # save Nii
            # saveToNii("predict_{}".format(img_name), val_outputs_onehot[0, :].cpu(), rot180=True, sitkcast=True,
            #           out_dir=args.niidir)

            organ_Dice_list = acc_func(val_outputs_onehot, val_labels_onehot[0:, 0:args.out_channels, :])
            organ_Dice_list = organ_Dice_list.cpu().numpy()[0]
            avg_acc_acc_org = np.nanmean([np.nanmean(l) for l in organ_Dice_list])
            print(organ_Dice_list)
            # print(organ_Dice_list.mean())
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
                # organ_Dice = dice(val_o, val_l)
                organ_Dice = organ_Dice_list[i - 1]
                uni = np.unique(val_l)
                if len(uni) > 1:
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
                csv_head.append(str(i) + "_Dice_organ")
                csv_head.append(str(i) + "_HD_organ")
                csv_head.append(str(i) + "_IOU_organ")
                csv_head.append(str(i) + "_precision_organ")
                csv_head.append(str(i) + "_recall_organ")
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

            mean_list_case.append([mean_dice, mean_HD, mean_IOU, mean_precision, mean_recall])
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

        csv_data_name.append("Mean")
        csv_data = []
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

        # for k in range(len(csv_data_all)):
        #     print(len(csv_data_all[k]))
        csv_data_all = np.array(csv_data_all)
        csv_dict = {}
        # csv_dict["method"] = 'unetr'
        csv_dict["sampleName"] = csv_data_name
        for j in range(len(csv_head)):
            csv_dict[csv_head[j]] = csv_data_all[:, j]
        df = pd.DataFrame(csv_dict)
        # 保存 dataframe
        df.to_csv(args.logfile + "-1.csv")

        # csv for report
        csv_dict = {}
        coefficient = ["dice", "HD95", "IOU", "precision", "recall"]
        # organKeys = np.array(organKeys[0:13])#BTCV 13 organs
        organKeys = np.append(organKeys, "Mean")

        col1 = []
        col2 = []
        col3 = []
        col4 = []
        col5 = []
        col6 = []
        col7=[]#dataset
        for m in range(len(csv_data_name)):  # 1mean + length of samples
            for j in range(args.out_channels):  # 1mean + length of organs
                for i in range(5):  # five coefficient
                    col1.append(args.method) #method
                    col2.append(csv_data_name[m])
                    col3.append(coefficient[i])
                    col4.append(str(j + 1))
                    col5.append(organKeys[j].replace("\n", ""))
                    col6.append(csv_data_all[m][i + 5 * j])
                    if csv_data_name[m][0:4]=="ABD_":
                        col7.append("1")
                    elif csv_data_name[m][0:8]=="Patient_":
                        col7.append("2")
                    elif csv_data_name[m][0:7]=="spleen_":
                        col7.append("3")
                    elif csv_data_name[m][0:9]=="pancreas_":
                        col7.append("4")
                    elif csv_data_name[m][0:7]=="spleen_":
                        col7.append("5")
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
        df.to_csv(args.logfile + "-report.csv")

if __name__ == "__main__":
        main()
