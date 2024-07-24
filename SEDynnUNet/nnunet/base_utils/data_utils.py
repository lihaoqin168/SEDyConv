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

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
from base_utils.utils import *
from batchgenerators.utilities.file_and_folder_operations import *

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    """one can create a class with a __call__ function that 
    calls your pre-processing functions taking into account that 
    not all of them are called on the labels"""
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),#normalizing the orientations of images
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # transforms.RandRotated(keys=["image", "label"], range_x=0.2, keep_size=True, prob=args.RandFlipd_prob),#add
            # transforms.RandRotated(keys=["image", "label"], range_y=0.2, keep_size=False, prob=args.RandFlipd_prob),#add
            # transforms.RandRotated(keys=["image", "label"], range_z=0.2, keep_size=False, prob=args.RandFlipd_prob),#add
            #
            # transforms.RandRotated(keys=["image", "label"], range_x=0.2, range_y=0.2, range_z=0.2, keep_size=False, prob=0.9),#add
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
                # allow_smaller=True
            ),
            # transforms.Resized(keys=["image", "label"], spatial_size=[args.roi_x, args.roi_y, args.roi_z],
            #                    mode="trilinear"),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=[0,1]),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            # transforms.Flipd(keys=["image", "label"], spatial_axis=0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "test", base_dir=data_dir)
        print("+ test_files", test_files)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            # persistent_workers=True
        )
        loader = test_loader
    elif args.valid_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            # persistent_workers=True
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=args.cache_num, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=my_collate
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            # persistent_workers=True
        )
        loader = [train_loader, val_loader]

    return loader

from monai.data.utils import list_data_collate
def my_collate(batch):
    dict = list_data_collate(batch)
    x = dict["label"]
    organsKeys = []
    for i in range(x.shape[0]):
        a = torch.index_select(x, 0, torch.tensor([i]))
        organsKeys.append(torch.unique(a).numpy())
    dict["organsKeys"] = organsKeys
    # for idx_list_batch in batch:
    #     for dict in idx_list_batch:
    #         print(torch.unique(dict["label"]))
    return dict

# def my_test_collate(batch):
#     dict = list_data_collate(batch)
#     x = dict["label"]
#     organsKeys = torch.ones((x.shape[0], args.out_channels))
#     dict["organsKeys"] = organsKeys
#     return dict



# from monai.transforms.inverse import InvertibleTransform
# from monai.transforms.transform import MapTransform
# from monai.transforms.spatial.array import Resize
#
# class fitROISize_Transform(MapTransform, InvertibleTransform):
#     """
#     Dictionary-based wrapper of :py:class:`monai.transforms.Resize`.
#
#     Args:
#         keys: keys of the corresponding items to be transformed.
#             See also: :py:class:`monai.transforms.compose.MapTransform`
#         spatial_size: expected shape of spatial dimensions after resize operation.
#             if some components of the `spatial_size` are non-positive values, the transform will use the
#             corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
#             to `(32, 64)` if the second spatial dimension size of img is `64`.
#         size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
#             if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
#             which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
#             https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
#             #albumentations.augmentations.geometric.resize.LongestMaxSize.
#         mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
#             The interpolation mode. Defaults to ``"area"``.
#             See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
#             It also can be a sequence of string, each element corresponds to a key in ``keys``.
#         align_corners: This only has an effect when mode is
#             'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
#             See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
#             It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
#         allow_missing_keys: don't raise exception if key is missing.
#     """
#
#     backend = Resize.backend
#
#     def __init__(
#         self,
#         keys: KeysCollection,
#         spatial_size: Union[Sequence[int], int],
#         size_mode: str = "all",
#         mode: InterpolateModeSequence = InterpolateMode.BILINEAR,
#         align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
#         allow_missing_keys: bool = False,
#     ) -> None:
#         super().__init__(keys, allow_missing_keys)
#         self.mode = ensure_tuple_rep(mode, len(self.keys))
#         self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
#         self.resizer = Resize(spatial_size=spatial_size, size_mode=size_mode)
#
#     def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
#         d = dict(data)
#         for key, mode, align_corners in self.key_iterator(d, self.mode, self.align_corners):
#             self.push_transform(
#                 d,
#                 key,
#                 extra_info={
#                     "mode": mode.value if isinstance(mode, Enum) else mode,
#                     "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
#                 },
#             )
#             d[key] = self.resizer(d[key], mode=mode, align_corners=align_corners)
#         return d
#
#     def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
#         d = deepcopy(dict(data))
#         for key in self.key_iterator(d):
#             transform = self.get_most_recent_transform(d, key)
#             orig_size = transform[TraceKeys.ORIG_SIZE]
#             mode = transform[TraceKeys.EXTRA_INFO]["mode"]
#             align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
#             # Create inverse transform
#             inverse_transform = Resize(
#                 spatial_size=orig_size,
#                 mode=mode,
#                 align_corners=None if align_corners == TraceKeys.NONE else align_corners,
#             )
#             # Apply inverse transform
#             d[key] = inverse_transform(d[key])
#             # Remove the applied transform
#             self.pop_transform(d, key)
#
#         return d
