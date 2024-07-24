

from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from base_utils.utils import nnUNet_raw_data
from base_utils.nii_utils import do_split

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
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_folder = join(base, "Training/img")
    label_folder = join(base, "Training/label")
    test_folder = join(base, "Testing/img")
    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(train_folder, join=False, suffix = 'nii.gz')
    for p in train_patients:
        serial_number = int(p[3:7])
        train_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        label_file = join(label_folder, f'label{p[3:]}')
        image_file = join(train_folder, p)
        shutil.copy(image_file, join(imagestr, f'{train_patient_name[:7]}_0000.nii.gz'))
        shutil.copy(label_file, join(labelstr, train_patient_name))
        train_patient_names.append(train_patient_name)

    test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    for p in test_patients:
        p = p[:-7]
        image_file = join(test_folder, p + ".nii.gz")
        serial_number = int(p[3:7])
        test_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        shutil.copy(image_file, join(imagests, f'{test_patient_name[:7]}_0000.nii.gz'))
        test_patient_names.append(test_patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "AbdominalOrganSegmentation"
    json_dict['description'] = "Multi-Atlas Labeling Beyond the Cranial Vault Abdominal Organ Segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "https://www.synapse.org/#!Synapse:syn3193805/wiki/217789"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "00": "background",
        "01": "spleen",
        "02": "right kidney",
        "03": "left kidney",
        "04": "gallbladder",
        "05": "esophagus",
        "06": "liver",
        "07": "stomach",
        "08": "aorta",
        "09": "inferior vena cava",
        "10": "portal vein and splenic vein",
        "11": "pancreas",
        "12": "right adrenal gland",
        "13": "left adrenal gland"}
    )
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    #for nnUnet
    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % train_patient_name} for i, train_patient_name in enumerate(train_patient_names)]
    json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]
    save_json(json_dict, os.path.join(out_base, "dataset.json"))

    #for 5 cross valid
    #for organsKeymapping
    fold = 0
    tr_keys, val_keys = do_split(out_base, train_patient_names, fold)
    json_dict0 = json_dict
    json_dict0['organsKeyMapping'] = OrderedDict({
        "01": "1",
        "02": "2",
        "03": "3",
        "04": "4",
        "05": "5",
        "06": "6",
        "07": "7",
        "08": "8",
        "09": "9",
        "10": "10",
        "11": "11",
        "12": "12",
        "13": "13"}
    )
    json_dict0['test'] = ["./imagesTs/%s" % f'{test_patient_name[:7]}_0000.nii.gz' for test_patient_name in
                         test_patient_names]
    json_dict0["training"] = [
        {'image': "./imagesTr/%s" % f'{tr_keys_patient_name[:7]}_0000.nii.gz', "label": "./labelsTr/%s" % tr_keys_patient_name} for
        i, tr_keys_patient_name in enumerate(tr_keys)]
    json_dict0["validation"] = [
        {'image': "./imagesTr/%s" % f'{val_keys_patient_name[:7]}_0000.nii.gz', "label": "./labelsTr/%s" % val_keys_patient_name} for
        i, val_keys_patient_name in enumerate(val_keys)]
    save_json(json_dict0, os.path.join(out_base, "dataset_"+str(fold)+".json"))

    for i in range(4):
        fold = i+1
        tr_keys, val_keys = do_split(out_base, train_patient_names, fold)
        json_dict1 = json_dict0
        json_dict1['test'] = ["./imagesTs/%s" % f'{test_patient_name[:7]}_0000.nii.gz' for test_patient_name in
                             test_patient_names]
        json_dict1["training"] = [
            {'image': "./imagesTr/%s" % f'{tr_keys_patient_name[:7]}_0000.nii.gz', "label": "./labelsTr/%s" % tr_keys_patient_name} for
            i, tr_keys_patient_name in enumerate(tr_keys)]
        json_dict1["validation"] = [
            {'image': "./imagesTr/%s" % f'{val_keys_patient_name[:7]}_0000.nii.gz', "label": "./labelsTr/%s" % val_keys_patient_name} for
            i, val_keys_patient_name in enumerate(val_keys)]
        save_json(json_dict1, os.path.join(out_base, "dataset_"+str(fold)+".json"))
