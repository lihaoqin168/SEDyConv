import cv2

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for z in range(0, image.shape[2], stepSize[2]):
        for y in range(0, image.shape[3], stepSize[1]):
            for x in range(0, image.shape[4], stepSize[0]):
                # yield the current window
                yield (x, y, z, image[:,:, z:z + windowSize[2], y:y + windowSize[1], x:x + windowSize[0]])


# 返回滑动窗结果集合，本示例暂时未用到
def get_slice(image, stepSize, windowSize):
    slice_sets = []
    winW, winH, winZ = windowSize
    for (x, y, z, window) in sliding_window(image, stepSize=stepSize, windowSize=windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[4] != winW or window.shape[3] != winH or window.shape[2] != winZ:
            continue

        slice = image[:,:, z:z+winZ, y:y+winH, x:x+winW]
        slice_sets.append(slice)
    return slice_sets

if __name__ == '__main__':

    import torch
    image = torch.randn(2, 1, 24, 48, 48)
    # image = torch.randn(2, 1, 48, 96, 96)
    # image = torch.randn(2, 1, 48, 192, 192)
    # 自定义滑动窗口的大小
    w = image.shape[4]
    h = image.shape[3]
    z = image.shape[2]
    # 本代码将图片分为3×3，共九个子区域，winW, winH和stepSize可自行更改

    # image = torch.randn(2, 1, 24, 48, 48) -> torch.Size([2, 1, 24, 48, 24])
    (winW, winH, winZ) = (int(w/2),int(h),int(z))
    stepSize = (int(w/4), int(h), int(z))#cnt 3
    #
    # # image = torch.randn(2, 1, 48, 96, 96) -> torch.Size([2, 1, 24, 48, 48])
    # (winW, winH, winZ) = (int(w/2),int(h/2),int(z/2))
    # stepSize = (int(w/4), int(h/4), int(z/4))#cnt 26

    # # image = torch.randn(2, 1, 48, 192, 192) -> torch.Size([2, 1, 24, 48, 48])
    # (winW, winH, winZ) = (int(w/4),int(h/4),int(z/2))
    # stepSize = (int(w/6), int(h/6), int(z/4))#cnt 74

    # cnt = 0
    # for (x, y, z, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH, winZ)):
    #     # if the window does not meet our desired window size, ignore it
    #     if window.shape[4] != winW or window.shape[3] != winH or window.shape[2] != winZ:
    #         continue
    #
    #     slice = image[:,:, z:z+winZ, y:y+winH, x:x+winW]
    #     print('slice', slice.shape, cnt)
    #     cnt+=1

    slices = get_slice(image, stepSize=stepSize, windowSize=(winW, winH, winZ))
    print('slices', len(slices))
    print('slices.shape', slices[0].shape)

