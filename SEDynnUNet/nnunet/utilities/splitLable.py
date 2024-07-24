import numpy as np
"""
labelgroup is list with group label. for example:[[4],[1,2,3]],means lable 4 in group1, lable 1,2,3 in group2.
splitLable function : split target into 2 group target.
"""
def splitLable(target, labelgroup):
    # print("target[0].shape")
    # print(target[0].shape)
    res_target1 = []
    res_target2 = []
    res_sub_target = []
    for eachTg_idx in range(len(target)):#times is equal to num_pool
        eachTg = target[eachTg_idx]
        # print(type(eachTg))
        #each pool
        eachTgNp = eachTg.detach().numpy()
        for i in range(2):#each branch
            for lable_numidx in range(len(labelgroup[i])):
                lable_num = labelgroup[i][lable_numidx]
                splitTarget = np.where(eachTgNp == lable_num, eachTgNp, 0)
                if len(res_sub_target)==i:
                    res_sub_target.append(splitTarget)
                else:
                    res_sub_target[i] = res_sub_target[i]+splitTarget#np.bitwise_or(res_sub_target[i],splitTarget)

        res_target1.append(res_sub_target[0])
        res_target2.append(res_sub_target[1])
        res_sub_target = []

        # if len(np.array(eachTg)[1][0][0])==192:
        #     print("out eachTg")
        #     img = np.array(eachTg)
        #     img_mask1 = np.where((img == 2)|(img == 3)|(img == 1), img, 0)
        #     img_mask2 = np.where((img == 4), img, 0)
        #     ## save
        #     import SimpleITK as sitk
        #     out_file1 = "/data3/target-123.nii.gz"
        #     out_file2 = "/data3/target-4.nii.gz"
        #     # img = sitk.GetImageFromArray(img_mask1)
        #     print("img_mask1[1][0]")
        #     print(img_mask1[1][0].shape)
        #     img = sitk.GetImageFromArray(img_mask1[1][0])
        #     img2 = sitk.GetImageFromArray(img_mask2[1][0])
        #     sitk.WriteImage(img, out_file1)
        #     sitk.WriteImage(img2, out_file2)

    # for ttttt in res_target1:
    #     print("-2---")
    #     if len(np.array(ttttt)[1][0][0])==192:
    #         ## save
    #         import SimpleITK as sitk
    #         img = sitk.GetImageFromArray(ttttt[1][0])
    #         out_file = "/data3/target1.nii.gz"
    #         sitk.WriteImage(img, out_file)
    # print(len(res_target2))
    # for ttttt in res_target2:
    #     print("-3---")
    #     if len(np.array(ttttt)[1][0][0])==192:
    #         ## save
    #         import SimpleITK as sitk
    #         img = sitk.GetImageFromArray(ttttt[1][0])
    #         out_file = "/data3/target2.nii.gz"
    #         sitk.WriteImage(img, out_file)
    #         from PIL import Image
    #         idx=0
    #         for tg in np.array(ttttt)[1][0]:
    #             idx += 1
    #             im = Image.fromarray(tg)
    #             im.convert('RGB').save("/data3/test2/target"+str(idx)+".jpg")
    #         break

    return res_target1, res_target2

#if labelgroup=[[1,3],[2,4]], each group output shape [2,3,21,192,168], change the gt 3->2, 2->1, 4->2
def splitLable_mixlable(target, labelgroup):
    res_target1, res_target2 = splitLable(target, labelgroup)
    #len()=num_classes
    for i in range(len(res_target1)):
        res_target1_sub = res_target1[i]
        res_target2_sub = res_target2[i]
        for lable_numidx in range(len(labelgroup[0])):
            lable_num = labelgroup[0][lable_numidx]
            res_target1_sub = np.where(res_target1_sub == lable_num, lable_numidx+1, 0)

        for lable_numidx in range(len(labelgroup[1])):
            lable_num = labelgroup[1][lable_numidx]
            res_target2_sub = np.where(res_target2_sub == lable_num, lable_numidx+1, 0)
        res_target1[i] = res_target1_sub
        res_target2[i] = res_target2_sub
    return res_target1, res_target2