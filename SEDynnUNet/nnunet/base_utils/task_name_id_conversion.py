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



from base_utils.utils import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


def convert_id_to_task_name(task_id: int):
    startswith = "Task%03.0d" % task_id
    if nnUNet_raw_data is not None:
        candidates_raw = subdirs(nnUNet_raw_data, prefix=startswith, join=False)
    else:
        candidates_raw = []
    unique_candidates = np.unique(candidates_raw)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one task name found for task id %d. Please correct that. (I looked in the "
                           "following folders:\n%s" % (task_id, nnUNet_raw_data))
    if len(unique_candidates) == 0:
        raise RuntimeError("Could not find a task with the ID %d. Make sure the requested task ID exists and that "
                           "nnU-Net knows where raw and preprocessed data are located (see Documentation - "
                           "Installation). Here are your currently defined folders:"
                           "nnUNet_raw_data_base=%s\nIf something is not right, adapt your environemnt "
                           "variables." %
                           (task_id,
                            os.environ.get('nnUNet_raw_data_base') if os.environ.get('nnUNet_raw_data_base') is not None else 'None',
                            ))
    return unique_candidates[0]


def convert_task_name_to_id(task_name: str):
    assert task_name.startswith("Task")
    task_id = int(task_name[4:7])
    return task_id
