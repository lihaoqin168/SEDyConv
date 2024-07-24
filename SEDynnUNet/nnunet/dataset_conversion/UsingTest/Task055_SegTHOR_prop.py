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


def convert_for_submission(source_dir, target_dir):
    """
    I believe they want .nii, not .nii.gz
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    maybe_mkdir_p(target_dir)
    for f in files:
        img = sitk.ReadImage(join(source_dir, f))
        out_file = join(target_dir, f[:-7] + ".nii")
        sitk.WriteImage(img, out_file)



if __name__ == "__main__":
    base = "/data3/medicalDatasets/segTHOR"

    task_id = 55
    task_name = "SegTHOR"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    train_patients = subfolders(join(base, "train"), join=False)
    for p in train_patients:
        curr = join(base, "train", p)
        label_file = join(curr, "GT.nii.gz")
        try:
            imgsitk = sitk.ReadImage(label_file)
            vol_mask = sitk.GetArrayFromImage(imgsitk)
            # distinct_val = np.unique(vol_mask)
            cnt_list = []
            total_cnt = np.sum(vol_mask > 0)
            for val in range(5):
                if val == 0: continue
                cnt_list.append(round(np.sum(vol_mask == val) / total_cnt *100,2))
            train_patient_names.append(cnt_list)
        except Exception as e:
            print(e)

    test_patients = subfolders(join(base, "test8Samples"), join=False)
    for p in test_patients:
        curr = join(base, "test8Samples", p)
        label_file = join(curr, "GT.nii.gz")
        try:
            imgsitk = sitk.ReadImage(label_file)
            vol_mask = sitk.GetArrayFromImage(imgsitk)
            # distinct_val = np.unique(vol_mask)
            cnt_list = []
            total_cnt = np.sum(vol_mask > 0)
            for val in range(5):
                if val == 0: continue
                cnt_list.append(round(np.sum(vol_mask == val) / total_cnt * 100, 2))
            train_patient_names.append(cnt_list)
        except Exception as e:
            print(e)

    # convert list into DataFrame
    df = pd.DataFrame(train_patient_names).transpose()
    train_patient_names = df.to_numpy()
    csv_dict = {}
    for i in range(4):
        csv_dict['dataset'] = 'SegTHORCV'
        csv_dict[str(i+1)] = train_patient_names[i]
    df = pd.DataFrame(csv_dict)
    # 保存 dataframe
    df.to_csv("055_Organ_Prop.csv")