import os
import SimpleITK as sitk


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

    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])

    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol


if __name__ == '__main__':
    vols_path = ''
    outspacing_0 = 0.351563  # according to the SHANXI
    outspacing_1 = 0.351563

    I_vol_list = ['reg_ADC.nii.gz', 'reg_DWI_0.nii.gz', 'reg_DWI_800.nii.gz', 'reg_T1C.nii.gz', 'T2.nii.gz']

    if os.path.exists(vols_path):
        dir_name_list = os.listdir(vols_path)
        for nii_dir in dir_name_list:
            I_nii_dir_folder = vols_path + nii_dir
            vols_list = os.listdir(I_nii_dir_folder)

            for vol_name in vols_list:
                if vol_name in I_vol_list:
                    I_nii_dir = vols_path + nii_dir + '\\' + vol_name
                    resampled_nii_dir = vols_path + nii_dir + '\\resample_' + vol_name

                    vol_ori = sitk.Image(sitk.ReadImage(I_nii_dir))
                    vol_ori_spacing = vol_ori.GetSpacing()
                    outspacing_2 = vol_ori_spacing[2]
                    outspacing = [outspacing_0, outspacing_1, outspacing_2]

                    vol_resampled = resample(outspacing, vol_ori)
                    sitk.WriteImage(vol_resampled, resampled_nii_dir)
                    print('pid_', nii_dir, ' ', vol_name, ' transformed!')