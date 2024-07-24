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
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk

from base_utils.utils import nnUNet_raw_data
from base_utils.nii_utils import do_split

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

    task_id = 155
    task_name = "SegTHOR"

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

    train_patient_names = []
    test_patient_names = []
    train_patients = subfolders(join(base, "train"), join=False)
    for p in train_patients:
        curr = join(base, "train", p)
        label_file = join(curr, "GT.nii.gz")
        image_file = join(curr, p + ".nii.gz")
        # #flip
        # imgsitk = sitk.ReadImage(image_file)
        # imgsitk = sitk.Flip(imgsitk, sitk.VectorBool([False,True,False]))
        # imgsitk.SetDirection((1,0,0,0,1,0,0,0,1))
        # imgsitk.SetOrigin((0,0,0))
        # sitk.WriteImage(imgsitk, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        # imgsitk = sitk.ReadImage(label_file)
        # imgsitk = sitk.Flip(imgsitk, sitk.VectorBool([False,True,False]))
        # imgsitk.SetDirection((1,0,0,0,1,0,0,0,1))
        # imgsitk.SetOrigin((0,0,0))
        # sitk.WriteImage(imgsitk, join(labelstr, p + ".nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    test_patients = subfolders(join(base, "test4Samples"), join=False)
    for p in test_patients:
        curr = join(base, "test4Samples", p)
        label_file = join(curr, "GT.nii.gz")
        image_file = join(curr, p + ".nii.gz")
        # imgsitk = sitk.ReadImage(image_file)
        # imgsitk = sitk.Flip(imgsitk, sitk.VectorBool([False,True,False]))
        # imgsitk.SetDirection((1,0,0,0,1,0,0,0,1))
        # imgsitk.SetOrigin((0,0,0))
        # sitk.WriteImage(imgsitk, join(imagests, p + "_0000.nii.gz"))
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelsts, p + ".nii.gz"))
        test_patient_names.append(p)


    json_dict = OrderedDict()
    json_dict['name'] = "SegTHOR"
    json_dict['description'] = "SegTHOR"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "esophagus",
        "2": "heart",
        "3": "trachea",
        "4": "aorta",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))

    #for Transformer
    fold = 0
    tr_keys, val_keys = do_split(out_base, train_patient_names, fold, 10, 1234567890)
    json_dict0 = json_dict
    json_dict0['organsKeyMapping'] = OrderedDict({
        "1": "5",
        "2": "14",
        "3": "15",
        "4": "8"}
    )
    json_dict0['test'] = [{'image': "./imagesTs/%s" % f'{test_patient_name}_0000.nii.gz', "label": "./labelsTs/%s.nii.gz" % test_patient_name.split("/")[-1]} for test_patient_name in
     test_patient_names]
    json_dict0["training"] = [
        {'image': "./imagesTr/%s" % f'{tr_keys_patient_name}_0000.nii.gz', "label": "./labelsTr/%s.nii.gz" % tr_keys_patient_name} for
        i, tr_keys_patient_name in enumerate(tr_keys)]

    json_dict0["validation"] = [
        {'image': "./imagesTr/%s" % f'{val_keys_patient_name}_0000.nii.gz', "label": "./labelsTr/%s.nii.gz" % val_keys_patient_name} for
        i, val_keys_patient_name in enumerate(val_keys)]
    save_json(json_dict0, os.path.join(out_base, "dataset_0.json"))
