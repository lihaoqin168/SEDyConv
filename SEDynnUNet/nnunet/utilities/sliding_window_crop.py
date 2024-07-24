import cv2

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# 返回滑动窗结果集合，本示例暂时未用到
def get_slice(image, stepSize, windowSize):
    slice_sets = []
    for (x, y, window) in sliding_window(image, stepSize, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
            continue
        slice = image[y:y + windowSize[1], x:x + windowSize[0]]
        slice_sets.append(slice)
    return slice_sets

if __name__ == '__main__':
    image = cv2.imread('pd19.png')

    # 自定义滑动窗口的大小
    w = image.shape[1]
    h = image.shape[0]
    # 本代码将图片分为3×3，共九个子区域，winW, winH和stepSize可自行更改
    (winW, winH) = (int(w/2),int(h/2))
    stepSize = (int(w/4), int(h/4))
    cnt = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # since we do not have a classifier, we'll just draw the window
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1000)

        slice = image[y:y+winH,x:x+winW]
        cv2.namedWindow('sliding_slice',0)
        cv2.imshow('sliding_slice', slice)
        cv2.waitKey(1000)
        cnt = cnt + 1

