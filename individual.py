from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import numpy as np
# import scipy.ndimage
from sklearn import metrics
# from skimage.morphology import watershed
# from skimage.feature import peak_local_max
# from scipy import ndimage as ndi, ndimage
import skfuzzy
from scipy.signal import correlate2d


# 这里效果一般，高斯模糊后，closing再opening，opening的kernel越大，precision上升，recall下降
# 可以调用所有的库？


def get_region(img):
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)  # maxLoc is reversed index

    delta_h = img.shape[0] // 7
    delta_w = img.shape[1] // 10
    print(maxLoc, maxVal)
    h1 = maxLoc[1] - delta_h
    h2 = maxLoc[1] + delta_h
    w1 = maxLoc[0] - delta_w
    w2 = maxLoc[0] + delta_w
    return (h1, h2, w1, w2), img[h1:h2, w1:w2]


def calculate_performance(img, label):
    img = img.flatten()
    label = label.flatten()
    # print('accuracy: ', metrics.accuracy_score(label, img))
    precision = metrics.precision_score(label, img, average="binary")
    recall = metrics.recall_score(label, img, average="binary")
    f1 = 2 * precision * recall / (precision + recall)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1 measure: ', f1)
    with open('record.txt', 'a') as f:
        f.write(f'precision: {precision}\n')
        f.write(f'recall: {recall}\n')
        f.write(f'f1 measure: {f1}\n')


def transform_pixel(p, a, b, BG=False):
    if BG:  # BG, need modify later
        if p <= a:
            return 1
        elif a < p <= (a + b) / 2:
            return 1 - 2 * ((p - a) / (b - a)) ** 2
        elif (a + b) / 2 < p < b:
            return 2 * ((p - a) / (b - a)) ** 2
        else:
            return 0
    else:
        return np.exp(-((p - b) / a) ** 2)


def detect_circle(img):  # need to delete
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=30, param2=10,
                               minRadius=0, maxRadius=0)
    print('circles: ', circles)
    if circles is not None:
        for circle in circles[0]:
            img = cv2.circle(img.copy(), (int(circle[0]), int(circle[1])), int(circle[2]),  # 分别是x,y,r
                             (0, 0, 255), 2)  # 作图
        plt.imshow(img)
        # plt.show()


def fcm(img, n, size):
    img = img.copy().reshape(1, -1)
    _centroids, _u_orig, _, _, _, _, _ = skfuzzy.cluster.cmeans(img, n, 2, error=0.005, maxiter=1000)
    _u_orig = np.vstack([i for _, i in sorted(zip(_centroids, _u_orig))])  # sort the list based on center
    _centroids = sorted(_centroids[:, 0])
    # # for j in range(nb_of_center):  # 这里按center排序
    # #     fcm_roi[0, u_orig.argmax(axis=0) == j] = j * int(255 // (nb_of_center - 1))   # 注释掉相当于不对图片进行分层了
    for j in range(n):  # 这里间隔就不是等距的了
        # print(j, centroids[j].astype(np.uint8))
        img[0, _u_orig.argmax(axis=0) == j] = _centroids[j].astype(np.uint8)

    img = img.reshape(size)
    return _centroids, img


def post_processing(post_image):
    # calculate_performance(coarse_segment, bin_ground_truth)   # 85, 89
    plt.subplot(2, 2, 1)
    plt.imshow(post_image, 'gray')
    # 注意。。。。。这里修改了kernel，原来是25
    # 要么是kernel大了，要么是threshold取高了
    _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # to match the structure
    post_image = cv2.morphologyEx(post_image, cv2.MORPH_CLOSE, _kernel, iterations=2)
    plt.subplot(2, 2, 2)
    plt.imshow(post_image, 'gray')
    _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    post_image = cv2.morphologyEx(post_image, cv2.MORPH_OPEN, _kernel, iterations=2)
    plt.subplot(2, 2, 3)
    plt.imshow(post_image, 'gray')
    # plt.show()

    _contours, _ = cv2.findContours(post_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    _contour = sorted(_contours, reverse=True, key=lambda l: len(l))[0]
    convex_hull_image = np.zeros(post_image.shape, dtype=np.uint8)
    convex_hull_pts = cv2.convexHull(_contour)
    cv2.drawContours(convex_hull_image, _contour, 0, (1, 1, 1), 10)
    cv2.drawContours(convex_hull_image, [convex_hull_pts], 0, (1, 1, 1), -1)
    plt.subplot(2, 2, 4)
    plt.imshow(convex_hull_image, 'gray')

    # plt.show()
    return convex_hull_image


def find_od(img_path, ground_truth_path, index):
    image = cv2.imread(img_path, 1)
    _, _, red_image = cv2.split(image.copy())  # get red layer
    ground_truth = cv2.imread(ground_truth_path, 0)
    _, bin_ground_truth = cv2.threshold(ground_truth, thresh=1, maxval=1, type=cv2.THRESH_BINARY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # from openCV tutorial
    clahe_red_image = clahe.apply(red_image.copy())
    median_clahe_red_image = cv2.medianBlur(clahe_red_image, 9)  # 9*9 median filter

    # ---------------------------------FCM------------------------------
    ratio = 14
    small_size = (red_image.shape[0] // ratio, red_image.shape[1] // ratio)  # scale down
    median_clahe_red_image = cv2.resize(median_clahe_red_image, small_size[::-1])  # cv2 is column * row
    nb_of_center = 7
    _, fcm_median_clahe_red_image = fcm(median_clahe_red_image, nb_of_center, small_size)

    plt.subplot(2, 2, 1)
    plt.imshow(fcm_median_clahe_red_image)
    # plt.imshow(cv2.threshold(median_clahe_red_image, thresh=254, maxval=1, type=cv2.THRESH_BINARY)[1])
    # ------------------------------after fuzzy C-means------------------------------
    _, fcm_median_clahe_red_image = cv2.threshold(fcm_median_clahe_red_image,
                                                  thresh=fcm_median_clahe_red_image.max() - 1,
                                                  maxval=1, type=cv2.THRESH_BINARY)
    # kernel = np.ones((5, 5), np.uint8)
    # fcm_median_clahe_red_image = cv2.morphologyEx(fcm_median_clahe_red_image, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((5, 5), np.uint8)
    (_, _, _, maxLoc) = cv2.minMaxLoc(median_clahe_red_image)
    maxLoc = list(maxLoc)
    tmp = fcm_median_clahe_red_image.copy()
    x = y = radius = 0
    for i in range(25, 0, -1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i))
        fcm_median_clahe_red_image = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel, iterations=3)
        centers, _ = cv2.findContours(fcm_median_clahe_red_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        right_center = None
        if i == 1 and len(centers) == 1:
            right_center = sorted(centers, reverse=True, key=lambda l: len(l))[0]
        elif len(centers) == 1:   # opening too much
            continue
        elif len(centers) > 1:
            centers = sorted(centers, reverse=True, key=lambda l: len(l))

            for j in range(len(centers)):
                if maxLoc in centers[j][:, 0, :]:
                    right_center = centers[j]
                # print(i, j, len(centers), centers[1][:, 0, :], centers[0].shape)
                else:   # there are multiple maximal intensity, so check the biggest cluster first
                    for e in centers[j][:, 0, :]:
                        if median_clahe_red_image[e[1], e[0]] == median_clahe_red_image.max():
                            # print(i, j, median_clahe_red_image.max())
                            right_center = centers[j]
                            break
                if right_center is not None:
                    break

        if right_center is not None:
            # fcm_median_clahe_red_image *= 0     # empty
            # cv2.drawContours(fcm_median_clahe_red_image, right_center, -1, (1, 1, 1))
            (x, y), radius = cv2.minEnclosingCircle(right_center)
            break
    # fcm_median_clahe_red_image = cv2.dilate(fcm_median_clahe_red_image, np.ones((2, 2), np.uint8), iterations=1)
    # is this hard coding?  dilate(2)*1  0.96, 0.95

    fcm_median_clahe_red_image = cv2.resize(fcm_median_clahe_red_image, red_image.shape[::-1])  # 变回原来的大小
    # calculate_performance(fcm_median_clahe_red_image, bin_ground_truth)       # 这里效果是0.98，0.75
    # edge_map = cv2.dilate(fcm_median_clahe_red_image, np.ones((3, 3), np.uint8)) - fcm_median_clahe_red_image

    # ---------------------------用一个最小的圆去fit---------------------------
    plt.subplot(2, 2, 2)
    plt.imshow(fcm_median_clahe_red_image)
    # centers, _ = cv2.findContours(fcm_median_clahe_red_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # (x, y), radius = cv2.minEnclosingCircle(sorted(centers, reverse=True, key=lambda l: len(l))[0])  # get the biggest
    center = (int(x * ratio), int(y * ratio))
    radius = int(radius * ratio)
    # edge_map = cv2.circle(fcm_median_clahe_red_image.copy(), center, radius, (1, 1, 1), -1)
    # calculate_performance(edge_map, bin_ground_truth)
    # small.png: 效果显著上升 0.98，0.92    closing(10)*1 && opening(10)*3
    # 这里上面如果不做closing, precision能达到1, 但是recall只有0.82
    # 减少opening的kernel, 则precision下降, recall上升

    # -------------------------------now i have ROI----------------------
    cof = 1.3
    r = int(cof * radius)
    print(center, radius, cof, r)
    roi = red_image.copy()[center[1] - r: center[1] + r, center[0] - r: center[0] + r]
    # mask = np.zeros((2 * r, 2 * r))
    mask = cv2.circle(np.zeros(red_image.shape), center, r, (1, 1, 1), -1)
    mask = mask[center[1] - r: center[1] + r, center[0] - r: center[0] + r]
    roi = roi * mask        # very important
    plt.subplot(2, 2, 3)
    plt.imshow(roi)
    # plt.show()
    # return

    # ----------------------------stage 1 fuzzy modeling--------------------
    nb_of_center = 10
    centroids, fcm_roi = fcm(roi.copy(), nb_of_center, roi.shape)
    plt.subplot(2, 2, 4)
    plt.imshow(fcm_roi)
    plt.savefig(f'Image {index} p1')
    # plt.show()
    plt.close()
    FP_fcm_roi = np.zeros((roi.shape[0], roi.shape[1], nb_of_center))

    # 一共30个centroid，第一个‘0’作为BG
    # print(centroids, len(centroids))
    for i in range(nb_of_center):  # 要不要处理最后一个？        怎么用一个函数对所有
        for j in range(roi.shape[0]):
            for k in range(roi.shape[1]):
                if not i:
                    FP_fcm_roi[j, k, i] = transform_pixel(int(roi[j, k]), a=centroids[i], b=centroids[i + 1],
                                                          BG=True)
                else:
                    FP_fcm_roi[j, k, i] = transform_pixel(int(roi[j, k]), a=centroids[i] - centroids[i - 1],
                                                          b=centroids[i])

    before_filter = np.argmax(FP_fcm_roi, axis=2)
    # plt.imshow((before_filter * (254/nb_of_center)).astype(np.uint8))

    kernel_h = 0.2 * np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
    cor_FP_fcm_roi = np.zeros((roi.shape[0], roi.shape[1], nb_of_center))
    for i in range(nb_of_center):
        cor_FP_fcm_roi[:, :, i] = correlate2d(FP_fcm_roi[:, :, i], kernel_h, mode='same')

    cor_FP_fcm_roi = np.argmax(cor_FP_fcm_roi, axis=2)
    cor_FP_fcm_roi = (cor_FP_fcm_roi * (254/nb_of_center)).astype(np.uint8)

    _, bin_cor_FP_fcm_roi = cv2.threshold(cor_FP_fcm_roi, thresh=int(cor_FP_fcm_roi.max() * 0.6), maxval=1, type=cv2.THRESH_BINARY)
    # thre = int((cor_FP_fcm_roi.max() - cor_FP_fcm_roi.min()) * 0.6 + cor_FP_fcm_roi.min())
    # _, bin_cor_FP_fcm_roi = cv2.threshold(cor_FP_fcm_roi, thresh=thre, maxval=1, type=cv2.THRESH_BINARY)
    # _, bin_cor_FP_fcm_roi = cv2.threshold(roi, thresh=int(roi.max() * 0.9), maxval=1, type=cv2.THRESH_BINARY)
    # 这也能84，90。。。那我特么在做什么？

    # -------------------------------------------post-processing----------------------------------
    final_image = np.zeros(red_image.shape)
    final_image[center[1] - r: center[1] + r, center[0] - r: center[0] + r] = post_processing(bin_cor_FP_fcm_roi)
    plt.savefig(f'Image {index} p2')
    # plt.show()
    plt.close()
    with open('record.txt', 'a') as f:
        f.write(f'Image {index}\n')
    calculate_performance(final_image, bin_ground_truth)


if __name__ == '__main__':
    for i in range(1, 55):
        if i < 10:
            image_path = f'original_retinal_images/IDRiD_0{i}.jpg'
            mask_path = f'optic_disc_segmentation_masks/IDRiD_0{i}_OD.tif'
        else:
            image_path = f'original_retinal_images/IDRiD_{i}.jpg'
            mask_path = f'optic_disc_segmentation_masks/IDRiD_{i}_OD.tif'
        print(f'Image {i} : ')
        find_od(image_path, mask_path, i)
