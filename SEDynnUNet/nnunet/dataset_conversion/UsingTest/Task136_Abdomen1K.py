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
from nnunet.dataset_conversion import resampleVolume, resampleLabelVolume


if __name__ == "__main__":
    base = "/data5/03datasets/AbdomenCT 1K"

    task_id = 136
    task_name = "AbdomenCT1K"
    prefix = 'Case'
    resmaple_spacing = [1.5,1.5,2.0]
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

    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(label_folder, join=False, suffix = 'nii.gz')
    for p in train_patients:
        train_patient_name = p
        p = p[:-7]
        label_file = join(label_folder, train_patient_name)
        image_file = join(train_folder, p+'_0000.nii.gz')
        # shutil.copy(image_file, join(imagestr, p+'_0000.nii.gz'))
        # shutil.copy(label_file, join(labelstr, train_patient_name))
        train_patient_names.append(train_patient_name)

        # try:
        #     # imgsitk = sitk.ReadImage(image_file)
        #     # imgsitk = resampleVolume.resample(resmaple_spacing, sitk.Image(imgsitk))
        #     # sitk.WriteImage(imgsitk, join(imagestr, p+'_0000.nii.gz'))
        #     imgsitk = sitk.ReadImage(label_file)
        #     imgsitk = resampleLabelVolume.resample(resmaple_spacing, sitk.Image(imgsitk))
        #     sitk.WriteImage(imgsitk, join(labelstr, train_patient_name))
        # except Exception:
        #     print(Exception)

    test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    for p in test_patients:
        test_patient_name = p
        p = p[:-7]
        label_file = join(test_folder, p + ".nii.gz")
        image_file = join(testimg_folder, p + "_0000.nii.gz")
        # shutil.copy(image_file, join(imagests, p+'_0000.nii.gz'))
        # shutil.copy(label_file, join(labelsts, test_patient_name))
        test_patient_names.append(test_patient_name)
        # try:
        #     # imgsitk = sitk.ReadImage(image_file)
        #     # imgsitk = resampleVolume.resample(resmaple_spacing, sitk.Image(imgsitk))
        #     # sitk.WriteImage(imgsitk, join(imagests, p+'_0000.nii.gz'))
        #     imgsitk = sitk.ReadImage(label_file)
        #     imgsitk = resampleLabelVolume.resample(resmaple_spacing, sitk.Image(imgsitk))
        #     sitk.WriteImage(imgsitk, join(labelsts, test_patient_name))
        # except Exception:
        #     print(Exception)

    json_dict = OrderedDict()
    json_dict['name'] = "AbdomenCT-1K"
    json_dict['description'] = "a large and diverse abdominal CT organ segmentation dataset with 1000+ CT scans"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = ""
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "00": "background",
        "01": "liver",
        "02": "kidney",
        "03": "spleen",
        "04": "pancreas",}
    )
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % train_patient_name} for i, train_patient_name in enumerate(train_patient_names)]
    json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))

    #for Unetr
    from base_utils.nii_utils import do_split
    #using splits_final.pkl or create 5-fold
    tr_keys, val_keys = do_split(out_base, train_patient_names, 0, 8, 12345678)
    #save json
    json_dict0 = json_dict

    json_dict0['test'] = [{'image': "./imagesTs/%s" % f'{test_patient_name[:-7]}_0000.nii.gz', "label": "./labelsTs/%s" % test_patient_name.split("/")[-1]} for test_patient_name in
     test_patient_names]
    json_dict0["training"] = [
        {'classify':'0,0','dsOrgKeys':'0','image': "./imagesTr/%s" % f'{tr_keys_patient_name[:-7]}_0000.nii.gz', "label": "./labelsTr/%s" % tr_keys_patient_name} for
        i, tr_keys_patient_name in enumerate(tr_keys)]

    json_dict0["validation"] = [
        {'image': "./imagesTr/%s" % f'{val_keys_patient_name[:-7]}_0000.nii.gz', "label": "./labelsTr/%s" % val_keys_patient_name} for
        i, val_keys_patient_name in enumerate(val_keys)]
    save_json(json_dict0, os.path.join(out_base, "dataset_0.json"))