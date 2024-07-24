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
from nnunet.dataset_conversion import resampleVolume
from nnunet.dataset_conversion import resampleLabelVolume


if __name__ == "__main__":
    base = "/data6/00dataset/amos22/amos22"

    task_id = 135
    task_name = "AMOS"
    prefix = 'amos'
    resmaple_spacing = [1.5,1.5,2]
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

    train_folder = join(base, "imagesTr")
    label_folder = join(base, "labelsTr")
    testimg_folder = join(base, "imagesTs72")
    test_folder = join(base, "labelsTs72")

    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(train_folder, join=False, suffix = 'nii.gz')
    for p in train_patients:
        train_patient_name = p
        p = p[:-7]
        # if p == 'amos_0115':
        if p != None:
            label_file = join(label_folder, train_patient_name)
            image_file = join(train_folder, p+'.nii.gz')
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
            # except Exception as e:
            #     print(e)

    test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    for p in test_patients:
        test_patient_name = p
        p = p[:-7]
        label_file = join(test_folder, p + ".nii.gz")
        image_file = join(testimg_folder, p + ".nii.gz")
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
        # except Exception as e:
        #     print(e)

    json_dict = OrderedDict()
    json_dict['name'] = "AMOS"
    json_dict['description'] = "AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "SRIDB x CUHKSZ x HKU x LGCHSZ x LGPHSZ"
    json_dict['licence'] = "CC-BY-SA 4.0"
    json_dict['release'] = "1.0 01/05/2022"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "0": "background",
        "1": "spleen",
        "2": "right kidney",
        "3": "left kidney",
        "4": "gall bladder",
        "5": "esophagus",
        "6": "liver",
        "7": "stomach",
        "8": "arota",
        "9": "postcava",
        "10": "pancreas",
        "11": "right adrenal gland",
        "12": "left adrenal gland",
        "13": "duodenum",
        "14": "bladder",
        "15": "prostate/uterus",}
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