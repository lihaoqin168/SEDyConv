import os
import SimpleITK as sitk
import numpy as np


def resample(outspacing, vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    # 读取文件的size和spacing信息
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()
    volOrigin = vol.GetOrigin()
    volDirection = vol.GetDirection()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])

    transform = sitk.Transform()
    transform.SetIdentity()
    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(volOrigin)
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(volDirection)
    resampler.SetSize(outsize)

    vol_mask = sitk.GetArrayFromImage(vol)
    distinct_val = np.unique(vol_mask)
    for val in distinct_val:
        if val == 0:
            newvol = np.zeros((outsize[2],outsize[1],outsize[0]))
            continue
        img_mask = np.where(vol_mask == val, 1, 0)
        # a = np.where(vol_mask == val)
        img = sitk.GetImageFromArray(img_mask)
        img.CopyInformation(vol)
        img = sitk.Cast(img, sitk.sitkUInt8)
        newimg = resampler.Execute(img)
        newimg_mask = sitk.GetArrayFromImage(newimg)
        #
        newimg_mask_cover = np.where(newimg_mask == 1, 0, 1)
        newimg_mask = np.where(newimg_mask == 1, val, 0)
        newvol = newvol * newimg_mask_cover
        newvol = np.add(newvol,newimg_mask)
        # b = np.where(newvol == val)
    img = sitk.GetImageFromArray(newvol)
    img.CopyInformation(newimg)
    newvol = sitk.Cast(img, sitk.sitkUInt8)
    return newvol


if __name__ == '__main__':
    vols_path = './test_nii/'
    outspacing_0 = 1.5  # according to the SHANXI
    outspacing_1 = 1.5

    I_vol_list = ['amos_0005.nii.gz','amos_0030.nii.gz']

    if os.path.exists(vols_path):
        dir_name_list = os.listdir(vols_path)
        for I_nii_dir in dir_name_list:
            vol_ori = sitk.Image(sitk.ReadImage(vols_path+I_nii_dir))
            vol_ori_spacing = vol_ori.GetSpacing()
            outspacing_2 = vol_ori_spacing[2]
            outspacing = [outspacing_0, outspacing_1, outspacing_2]

            vol_resampled = resample(outspacing, vol_ori)
            sitk.WriteImage(vol_resampled, vols_path+'resample_'+I_nii_dir)
            print('pid_', I_nii_dir, ' transformed!')