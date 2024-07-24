import numpy as np

def splitLable(target, splitlabel):
    # target map(shape(b, 1, x, y(, z))
    # splitlabel = [[1], [2, 3, 4]]
    print(splitlabel)
    res_target1 = []
    res_target2 = []
    res_sub_target = []
    for eachTg_idx in range(len(target)):#times is equal to num_pool
        eachTg = target[eachTg_idx]
        #each pool
        # eachTgNp = np.array(eachTg)
        eachTgNp = eachTg.detach().numpy()
        print("eachTgNp.shape")
        print(eachTgNp.shape)
        for i in range(2):
            #each branch
            for lable_numidx in range(len(splitlabel[i])):
                lable_num = splitlabel[i][lable_numidx]
                splitTarget = np.where(eachTgNp == lable_num, eachTgNp, 0)
                print("i="+str(i)+"   res_sub_target:"+str(len(res_sub_target)))
                if len(res_sub_target)==i:
                    print("lable_num:"+str(lable_num))
                    res_sub_target.append(splitTarget)
                else:
                    print(str(i)+"np.bitwise_or lable_num:"+str(lable_num))
                    res_sub_target[i] = res_sub_target[i]+splitTarget#np.bitwise_or(res_sub_target[i],splitTarget)

        print("res_sub_target[0].shape")
        print(res_sub_target[0].shape)
        res_target1.append(res_sub_target[0])
        res_target2.append(res_sub_target[1])
        res_sub_target = []

        print("-1---")
        print(type(eachTg))
        print(len(np.array(eachTg)[1][0][0]))
        if len(np.array(eachTg)[1][0][0])==192:
            print("out eachTg")
            img = np.array(eachTg)
            img_mask1 = np.where((img == 2)|(img == 3)|(img == 1), img, 0)
            img_mask2 = np.where((img == 4), img, 0)
            ## save
            import SimpleITK as sitk
            out_file1 = "/data3/target-123.nii.gz"
            out_file2 = "/data3/target-4.nii.gz"
            # img = sitk.GetImageFromArray(img_mask1)
            print("img_mask1[1][0]")
            print(img_mask1[1][0].shape)
            img = sitk.GetImageFromArray(img_mask1[1][0])
            img2 = sitk.GetImageFromArray(img_mask2[1][0])
            sitk.WriteImage(img, out_file1)
            sitk.WriteImage(img2, out_file2)
            from PIL import Image
            idx = 0
            for tg in img_mask1[1][0]:
                idx += 1
                im = Image.fromarray(tg)
                im.convert('RGB').save("/data3/test/origin_target" + str(idx) + ".jpg")
    """
    """
    print("len(res_target1)")
    print(len(res_target1))
    for ttttt in res_target1:
        print("-2---")
        print(len(np.array(ttttt)[1][0][0]))
        if len(np.array(ttttt)[1][0][0])==192:
            print("out tttt" + str(ttttt[1][0].shape))
            ## save
            import SimpleITK as sitk
            img = sitk.GetImageFromArray(ttttt[1][0])
            out_file = "/data3/target1.nii.gz"
            sitk.WriteImage(img, out_file)
    print(len(res_target2))
    for ttttt in res_target2:
        print("-3---")
        print(len(np.array(ttttt)[1][0][0]))
        if len(np.array(ttttt)[1][0][0])==192:
            print("out tttt" + str(ttttt[1][0].shape))
            ## save
            import SimpleITK as sitk
            img = sitk.GetImageFromArray(ttttt[1][0])
            out_file = "/data3/target2.nii.gz"
            sitk.WriteImage(img, out_file)
            from PIL import Image
            idx=0
            for tg in np.array(ttttt)[1][0]:
                idx += 1
                im = Image.fromarray(tg)
                im.convert('RGB').save("/data3/test2/target"+str(idx)+".jpg")
            break

    return target, target



