import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

img_L = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
img_R = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
n = sys.argv[3]
m = sys.argv[4]

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Find the keypoints and descriptors with SIFT for each img
keypoints_L, descripoints_L = sift.detectAndCompute(img_L, None)
keypoints_R, descripoints_R = sift.detectAndCompute(img_R, None)


def sort_by_response(kp):
    sort_kp = sorted(kp, key=lambda point: point.response)
    return sort_kp


# Saving last n key-points of the sorted list for get the n strongest key-points
sort_kp_L = sort_by_response(keypoints_L)[-int(n):-1]
sort_kp_R = sort_by_response(keypoints_R)[-int(n):-1]


# Return fit descripoints for group of keypoint
def sub_descriptors(part_kp, full_kp, full_des):
    part_des = []
    for point in part_kp:
        for i in range(len(full_kp)):
            if point == full_kp[i]:
                part_des.append(full_des[i])
    return np.asarray(part_des)


sub_des_R = sub_descriptors(sort_kp_R, keypoints_R, descripoints_R)


def click_p(event, x, y, flags, param, radius=4):
    if event == cv2.EVENT_LBUTTONDOWN:
        point = None

        # Check if the click (x,y) is close enough to keypoint coordinates.
        for p in sort_kp_L:
            if (abs(y - p.pt[1]) + abs(x - p.pt[0])) < radius:
                    point = p
                    break
        if point is not None:
            single = [point]
            sub_des_L = sub_descriptors(single, keypoints_L, descripoints_L)

            # BFMatcher with default params
            bf1 = cv2.BFMatcher()
            matches = bf1.knnMatch(sub_des_L, sub_des_R, k=int(m))

            # Saving the matches m points in a list
            match_points_list = []
            for mat in matches[0]:
                img2_idx = mat.trainIdx
                (x2, y2) = sort_kp_R[img2_idx].pt
                match_points_list.append((x2, y2))

            # Writing the m points to a text file
            file = open('display.txt', 'w+')
            file.write('The selected point coordinates:\n')
            file.write('(' + str(x) + ',' + str(y) + ')\n')
            file.write('The m matching points:\n')
            for i in match_points_list:
                x_str = str(i[0])
                y_str = str(i[1])
                to_write = '('+x_str+','+y_str+')'
                file.write(to_write+'\n')

            good = []
            for i in matches[0]:
                good.append([i])

            # cv2.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(img_L, single, img_R, sort_kp_R, good, outImg=img_L)
            cv2.imshow("output", img3)


img_L = cv2.drawKeypoints(img_L, sort_kp_L, None)
cv2.imshow("Image", img_L)
img_R = cv2.drawKeypoints(img_R, sort_kp_R, None)
cv2.imshow("Image!", img_R)

cv2.setMouseCallback('Image', click_p)

cv2.waitKey(0)
cv2.destroyAllWindows()
