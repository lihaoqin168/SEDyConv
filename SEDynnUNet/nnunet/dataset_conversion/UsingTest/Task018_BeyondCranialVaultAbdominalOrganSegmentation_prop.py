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
import SimpleITK as sitk
import pandas as pd
import numpy as np
import shutil


if __name__ == "__main__":
    base = "/data3/medicalDatasets/Adomen/RawData/"

    task_id = 18
    task_name = "AbdominalOrganSegmentation"
    prefix = 'ABD'

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

    train_folder = join(base, "Training/img")
    label_folder = join(base, "Training/label")
    test_folder = join(base, "Testing6Samples/img")
    testlabel_folder = join(base, "Testing6Samples/label")
    train_patient_names = []
    train_patients = subfiles(train_folder, join=False, suffix = 'nii.gz')
    for p in train_patients:
        serial_number = int(p[3:7])
        train_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        label_file = join(label_folder, f'label{p[3:]}')
        try:
            imgsitk = sitk.ReadImage(label_file)
            vol_mask = sitk.GetArrayFromImage(imgsitk)
            # distinct_val = np.unique(vol_mask)
            cnt_list = []
            total_cnt = np.sum(vol_mask > 0)
            for val in range(14):
                if val == 0: continue
                cnt_list.append(round(np.sum(vol_mask == val) / total_cnt *100,2))
            train_patient_names.append(cnt_list)
        except Exception as e:
            print(e)

    test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    for p in test_patients:
        serial_number = int(p[3:7])
        label_file = join(testlabel_folder, f'label{p[3:]}')
        try:
            imgsitk = sitk.ReadImage(label_file)
            vol_mask = sitk.GetArrayFromImage(imgsitk)
            # distinct_val = np.unique(vol_mask)
            cnt_list = []
            total_cnt = np.sum(vol_mask > 0)
            for val in range(14):
                if val == 0: continue
                cnt_list.append(round(np.sum(vol_mask == val) / total_cnt *100,2))
            train_patient_names.append(cnt_list)
        except Exception as e:
            print(e)
    # convert list into DataFrame
    df = pd.DataFrame(train_patient_names).transpose()
    train_patient_names = df.to_numpy()
    csv_dict = {}
    for i in range(13):
        csv_dict['dataset'] = 'BTCV'
        csv_dict[str(i+1)] = train_patient_names[i]
    df = pd.DataFrame(csv_dict)
    # 保存 dataframe
    df.to_csv("018_Organ_Prop.csv")