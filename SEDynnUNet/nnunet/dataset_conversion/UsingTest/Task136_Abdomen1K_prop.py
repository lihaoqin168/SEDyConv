#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    base = "/data5/03datasets/AbdomenCT 1K"

    task_id = 136
    task_name = "AbdomenCT1K"
    prefix = 'Case'
    # resmaple_spacing = [0.80000001,0.79101562,0.79101562]
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_folder = join(base, "AbdomenCT-1K-Image")
    label_folder = join(base, "Mask")
    test_folder = join(base, "testMask")
    testimg_folder = join(base, "AbdomenCT-1K-Image")

    max_XYZspacing=[0,0,0,0,0,0]
    min_XYZspacing = [9999,9999,9999,9999,9999,9999]

    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(label_folder, join=False, suffix = 'nii.gz')
    for p in train_patients:
        train_patient_name = p
        p = p[:-7]
        if p != None:
            label_file = join(label_folder, train_patient_name)
            try:
                imgsitk = sitk.ReadImage(label_file)

                s = imgsitk.GetSpacing()
                w = imgsitk.GetWidth()
                h = imgsitk.GetHeight()
                d = imgsitk.GetDepth()
                if max_XYZspacing[0]<s[0]:
                    max_XYZspacing[0]=s[0]
                if max_XYZspacing[1]<s[1]:
                    max_XYZspacing[1]=s[1]
                if max_XYZspacing[2]<s[2]:
                    max_XYZspacing[2]=s[2]
                if max_XYZspacing[3]<w:
                    max_XYZspacing[3]=w
                if max_XYZspacing[4]<h:
                    max_XYZspacing[4]=h
                if max_XYZspacing[5]<d:
                    max_XYZspacing[5]=d

                if min_XYZspacing[0]>s[0]:
                    min_XYZspacing[0]=s[0]
                if min_XYZspacing[1]>s[1]:
                    min_XYZspacing[1]=s[1]
                if min_XYZspacing[2]>s[2]:
                    min_XYZspacing[2]=s[2]
                if min_XYZspacing[3]>w:
                    min_XYZspacing[3]=w
                if min_XYZspacing[4]>h:
                    min_XYZspacing[4]=h
                if min_XYZspacing[5]>d:
                    min_XYZspacing[5]=d

                vol_mask = sitk.GetArrayFromImage(imgsitk)
                cnt_list = []
                total_cnt = np.sum(vol_mask>0)
                for val in range(16):
                    if val==0:continue
                    cnt_list.append(round(np.sum(vol_mask == val) / total_cnt *100, 2))
                train_patient_names.append(cnt_list)
            except Exception as e:
                print(e)

    test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    for p in test_patients:
        test_patient_name = p
        p = p[:-7]
        label_file = join(test_folder, p + ".nii.gz")

        try:
            imgsitk = sitk.ReadImage(label_file)

            s = imgsitk.GetSpacing()
            w = imgsitk.GetWidth()
            h = imgsitk.GetHeight()
            d = imgsitk.GetDepth()
            if max_XYZspacing[0] < s[0]:
                max_XYZspacing[0] = s[0]
            if max_XYZspacing[1] < s[1]:
                max_XYZspacing[1] = s[1]
            if max_XYZspacing[2] < s[2]:
                max_XYZspacing[2] = s[2]
            if max_XYZspacing[3] < w:
                max_XYZspacing[3] = w
            if max_XYZspacing[4] < h:
                max_XYZspacing[4] = h
            if max_XYZspacing[5] < d:
                max_XYZspacing[5] = d

            if min_XYZspacing[0] > s[0]:
                min_XYZspacing[0] = s[0]
            if min_XYZspacing[1] > s[1]:
                min_XYZspacing[1] = s[1]
            if min_XYZspacing[2] > s[2]:
                min_XYZspacing[2] = s[2]
            if min_XYZspacing[3] > w:
                min_XYZspacing[3] = w
            if min_XYZspacing[4] > h:
                min_XYZspacing[4] = h
            if min_XYZspacing[5] > d:
                min_XYZspacing[5] = d

            vol_mask = sitk.GetArrayFromImage(imgsitk)
            # distinct_val = np.unique(vol_mask)
            cnt_list = []
            total_cnt = np.sum(vol_mask > 0)
            for val in range(16):
                if val == 0: continue
                cnt_list.append(round(np.sum(vol_mask == val) / total_cnt *100, 2))
            train_patient_names.append(cnt_list)
        except Exception as e:
            print(e)


    print('max_XYZspacing', max_XYZspacing)
    print('min_XYZspacing', min_XYZspacing)

    # # convert list into DataFrame
    # df = pd.DataFrame(train_patient_names).transpose()nnUNet_raw_data_base=/data4/nnUNet_raw_data_base
    # train_patient_names = df.to_numpy()
    # csv_dict = {}
    # for i in range(15):
    #     csv_dict['dataset'] = 'AbdomenCT1K'
    #     csv_dict[str(i+1)] = train_patient_names[i]
    # df = pd.DataFrame(csv_dict)
    # # 保存 dataframe
    # df.to_csv("136_Organ_Prop.csv")