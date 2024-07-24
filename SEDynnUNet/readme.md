Please read readme_unetr.md documents before read this file.

# SEDy-nnUNet

step 1. prepare dataset and conversion it (dataset_conversion.md)
 modify and run .py
 ```python ./nnunet/dataset_conversion/UsingTest/Task018_BeyondCranialVaultAbdominalOrganSegmentation.py```

step 2. preprocess
```nnUNet_plan_and_preprocess -t 018 -pl2d None```

step 3. modify short epochs

./nnunet/training/network_training/SEDynnUNetTrainerV2.py
```self.max_num_epochs = 500```

step 6. run:
```nnUNet_train 3d_fullres SEDynnUNetTrainerV2 018 0```

step 7. predict:
```nnUNet_predict --disable_tta  -i /data4/nnUNet_raw_data_base/nnUNet_raw_data/Task018_AbdominalOrganSegmentation/imagesTs/ -o /data3/predict_nii_nnUnet/018/SEDynnUNetTrainerV2-018best -t 018 -m 3d_fullres -tr SEDynnUNetTrainerV2 -f 0 -chk model_best```
```nnUNet_predict --disable_tta  -i /data4/nnUNet_raw_data_base/nnUNet_raw_data/Task018_AbdominalOrganSegmentation/imagesTs/ -o /data3/predict_nii_nnUnet/018/SEDynnUNetTrainerV2-018 -t 018 -m 3d_fullres -tr SEDynnUNetTrainerV2 -f 0```

step 8. evaluate:
```python export_execl_nii_calculate_metrics.py```

For more information about nnU-Net, please read the following paper:


    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
    for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

Please also cite this paper if you are using nnU-Net for your research!

