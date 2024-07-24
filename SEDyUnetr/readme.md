Please read readme_unetr.md documents before read this file.


# SEDy-unetr

step 1. prepare dataset

 ```python ./dataset/Task018_BeyondCranialVaultAbdominalOrganSegmentation.py```

step 2. train

train unetr

```python main_SEDyUnetr.py --model_name=unetr --res_block --num_samples=2 --logdir=unetr_task018 --feature_size=16 --batch_size=2 --cache_num=30  --out_channels=14 --max_epochs=2500 --json_list=dataset_0.json --val_every=50 --save_checkpoint --optim_lr=1e-5 --lrschedule=warmup_cosine --infer_overlap=0.5  --data_dir=/data4/dataset/Task018_AbdominalOrganSegmentation/```

train SEDyUnetr

```python main_SEDyUnetr.py --model_name=SEDyUnetr --temp_epoch=150 --res_block --num_samples=2 --logdir=SEDyUnetr_task018 --kernel_num=4  --feature_size=16 --batch_size=2 --cache_num=30  --out_channels=14 --max_epochs=2500 --json_list=dataset_0.json --val_every=50 --save_checkpoint --optim_lr=1e-5 --lrschedule=warmup_cosine --infer_overlap=0.5  --data_dir=/data4/dataset/Task018_AbdominalOrganSegmentation/```

step 3. predict and evaluate

train unetr

```python predict_unetr.py  --network_name=unetr --res_block --organs_file=./networks/organs13BTCV.key --mdir=/data3/SEDyUnetr/trained_models/ --pretrained_dir=unetr_task018 --logdir=/data3/predict_SEDyUnetr/runs/ --out_channels=14  --json_list=dataset_0.json --infer_overlap=0.5  --data_dir=/data4/dataset/Task018_AbdominalOrganSegmentation/```

train SEDyUnetr

```python predict_unetr.py  --network_name=SEDyUnetr --dy_flg --res_block --feature_size=16  --kernel_num=4 --organs_file=./networks/organs13BTCV.key --mdir=/data3/SEDyUnetr/trained_models/ --pretrained_dir=SEDyUnetr_task018 --logdir=/data3/predict_SEDyUnetr/runs/ --out_channels=14  --kernel_num=4 --json_list=dataset_0.json --infer_overlap=0.5  --data_dir=/data4/dataset/Task018_AbdominalOrganSegmentation/```


## Citation
If you find this repository useful, please consider citing UNETR paper:

```
@inproceedings{hatamizadeh2022unetr,
  title={Unetr: Transformers for 3d medical image segmentation},
  author={Hatamizadeh, Ali and Tang, Yucheng and Nath, Vishwesh and Yang, Dong and Myronenko, Andriy and Landman, Bennett and Roth, Holger R and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={574--584},
  year={2022}
}
```

## References
[1] Hatamizadeh, Ali, et al. "UNETR: Transformers for 3D Medical Image Segmentation", 2021. https://arxiv.org/abs/2103.10504.

[2] Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
", 2020. https://arxiv.org/abs/2010.11929.
