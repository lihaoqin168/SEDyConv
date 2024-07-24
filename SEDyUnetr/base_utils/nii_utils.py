
import os
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
from monai.utils import convert_to_tensor
from datetime import datetime
from base_utils.utils import print_to_log_file
from collections import OrderedDict
from sklearn.model_selection import KFold

def do_split(out_base, keys, fold, n_splits=5, seed=12345):
    """
            fold: a number of fold
            The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
            so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
            Sometimes you may want to create your own split for various reasons. You can create as many splits in this file as you want.
            If splits_pkl_directory is None, return a random validate
            :return: tr_keys, val_keys
            """
    timestamp = datetime.now()
    logfile = os.path.join(out_base,"do_split.log_%d_%d_%d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day))
    if out_base is not None:
        splits_file = os.path.join(out_base, "splits_final.pkl")

        # if the split file does not exist we need to create it
        if os.path.isfile(splits_file):
            print_to_log_file(logfile, "INFO: Using splits from existing split file:", splits_file)
            splits = load_pickle(splits_file)
            print_to_log_file(logfile, "INFO: The split file contains %d splits." % len(splits))
        else:
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                print_to_log_file(logfile, "INFO: Creating new %d-fold cross-validation split..." % n_splits)
                splits = []
                all_keys_sorted = np.sort(list(keys))
                kfold = KFold(n_splits, shuffle=True, random_state=seed)
                # kfold = KFold(n_splits=8, shuffle=True, random_state=12345678)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

        # print_to_log_file("Desired fold for training: %d" % self.fold)
        if fold < len(splits):
            tr_keys = splits[fold]['train']
            val_keys = splits[fold]['val']
            print_to_log_file(logfile, "INFO: This split has %d training and %d validation cases."
                              % (len(tr_keys), len(val_keys)))
        else:
            print_to_log_file(logfile, "INFO: now creating a random (but seeded) 80:20 split!")
            # if we request a fold that is not in the split file, create a random 80:20 split
            rnd = np.random.RandomState(seed=seed + fold)
            keys = np.sort(list(keys))
            idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
            idx_val = [i for i in range(len(keys)) if i not in idx_tr]
            tr_keys = [keys[i] for i in idx_tr]
            val_keys = [keys[i] for i in idx_val]
            print_to_log_file(logfile, "INFO: This random 80:20 split has %d training and %d validation cases."
                              % (len(tr_keys), len(val_keys)))
    return tr_keys,val_keys