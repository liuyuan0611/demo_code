"""
Identity card demo
Author: liuyuan
2018.2
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(src_image):
    shape = src_image.shape
    src_size = (shape[1], shape[0])
    new_size = (src_size[0]//4, src_size[1]//4)
    dst = cv2.resize(src_image, new_size)
    return dst


def find_contours(src_image):
    max_contour= None
    max_contour_len = 0
    result_image, contours, hierarchy = cv2.findContours(src_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        c = contours[i]
        contour_len = cv2.arcLength(c, True)
        if contour_len > max_contour_len:
            max_contour_len = contour_len
            max_contour = c
    return max_contour


def get_mask(src_image, contour):
    mask_image = np.zeros((src_image.shape[0], src_image.shape[1]), np.uint8)
    cv2.drawContours(mask_image, [contour], 0, (255, 255, 255), -1)
    return mask_image


def get_quads_points(contour):
    left_top = [2000, 2000]
    right_bottom = [0, 0]
    left_bottom = None
    right_top = None

    for i in range(contour.shape[0]):
        point = contour[i, 0, :]
        if point[0] < left_top[0] and point[1] < left_top[1]:
            left_top = point
        if point[0] > right_bottom[0] and point[1] > right_bottom[1]:
            right_bottom = point

    x_len = abs(right_bottom[0] - left_top[0])
    y_len = abs(right_bottom[1] - left_top[1])

    for i in range(contour.shape[0]):
        point = contour[i, 0, :]
        # skip left top point
        if point[0] == left_top[0] and point[1] == left_top[1]:
            continue
        # skip right bottom point
        if point[0] == left_top[0] and point[1] == left_top[1]:
            continue
        if abs(point[0] - left_top[0]) <= (0.5 * x_len) and abs(point[1] - right_bottom[1]) <= (0.5 * y_len):
            left_bottom = point
        if abs(point[0] - right_bottom[0]) <= (0.5 * x_len) and abs(point[1] - left_top[1]) <= (0.5 * y_len):
            right_top = point

    return left_top, right_top, right_bottom, left_bottom


def perspective_transform(src_image, left_top, right_top, right_bottom, left_bottom):
    ratio = 54 / 85.6
    x_len = right_bottom[0] - left_top[0]
    y_len = ratio * x_len
    src_points = np.float32([left_top, right_top, right_bottom, left_bottom])
    dst_points = np.float32([np.float32([left_bottom[0], left_top[1]]),
                             np.float32([right_bottom[0], left_top[1]]),
                             np.float32([right_bottom[0], left_top[1] + y_len]),
                             np.float32([left_bottom[0], left_top[1] + y_len])])
    print(src_points), print(dst_points)
    m = cv2.getPerspectiveTransform(src_points, dst_points)
    dst_image = cv2.warpPerspective(src_image, m, (src_image.shape[1], src_image.shape[0]))
    return dst_image


def get_result_image(foreground, background):
    for i in range(foreground.shape[0]):
        for j in range(foreground.shape[1]):
            if foreground[i, j, 0] == 0 and foreground[i, j, 1] == 0 and foreground[i, j, 2] == 0:
                background[i, j, :] = np.uint8([255, 255, 255])
    return background


def display_image(src_image, edges_image, mask_image, composite_image, trans_image, result_image):
    disp_src_image = src_image[:, :, ::-1]
    disp_comp_image = composite_image[:, :, ::-1]
    disp_trans_image = trans_image[:, :, ::-1]
    disp_result_image = result_image[:, :, ::-1]

    plt.subplot(231)
    plt.title('Source Image')
    plt.imshow(disp_src_image)

    plt.subplot(232)
    plt.title('Edges Image')
    plt.imshow(edges_image, cmap='gray')

    plt.subplot(233)
    plt.title('Mask Image')
    plt.imshow(mask_image, cmap='gray')

    plt.subplot(234)
    plt.title('Composite Image')
    plt.imshow(disp_comp_image)

    plt.subplot(235)
    plt.title('Transformed Image')
    plt.imshow(disp_trans_image)

    plt.subplot(236)
    plt.title('Result Image')
    plt.imshow(disp_result_image)

    plt.show()


def process(file_path):
    src = cv2.imread(file_path)
    resized = resize_image(src)
    source = resized.copy()

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 1]
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find edges using canny operator
    edges = cv2.Canny(thresh, 100, 200, apertureSize=3)
    # find contours
    contour = find_contours(edges)
    # find quad contour
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # draw contour
    cv2.drawContours(resized, [contour], -1, (0, 255, 0), 3)
    cv2.drawContours(resized, [approx], -1, (255, 0, 0), 3)
    left_top, right_top, right_bottom, left_bottom = get_quads_points(approx)
    # get mask area
    mask = get_mask(resized, contour)
    composite = cv2.bitwise_and(resized, resized, mask=mask)
    # perspective transform
    transform = perspective_transform(composite, left_top, right_top, right_bottom, left_bottom)

    foreground = cv2.bitwise_and(source, source, mask=mask)
    background = foreground.copy()
    result = get_result_image(foreground, background)
    result = perspective_transform(result, left_top, right_top, right_bottom, left_bottom)

    display_image(source, edges, mask, composite, transform, result)







