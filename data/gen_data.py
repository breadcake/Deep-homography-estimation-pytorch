import os
import cv2
import random
import numpy as np
from numpy.linalg import inv
import time

train_path = 'D:/Workspace/Datasets/coco2014/train2014'
val_path = 'D:/Workspace/Datasets/coco2014/val2014'
test_path = 'D:/Workspace/Datasets/coco2014/test2014'


def ImagePreProcessing(image_path, rho, patch_size, imsize):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, imsize)

    position_p = (random.randint(rho, imsize[0] - rho - patch_size), random.randint(rho, imsize[1] - rho - patch_size))
    tl_point = position_p
    tr_point = (patch_size + position_p[0], position_p[1])
    br_point = (patch_size + position_p[0], patch_size + position_p[1])
    bl_point = (position_p[0], patch_size + position_p[1])

    test_image = img.copy()
    four_points = [tl_point, tr_point, br_point, bl_point]

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)

    warped_image = cv2.warpPerspective(img, H_inverse, imsize)

    # Extract image patches (not stored)
    # Ip1 = test_image[tl_point[1]:br_point[1], tl_point[0]:br_point[0]]
    # Ip2 = warped_image[tl_point[1]:br_point[1], tl_point[0]:br_point[0]]

    training_image = np.dstack((img, warped_image))
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    datum = (training_image, np.array(four_points), H_four_points)

    return datum


# save .npy files
def savedata(source_path, new_path, rho, patch_size, imsize, data_size):
    lst = os.listdir(source_path + '/')
    filenames = [os.path.join(source_path, l) for l in lst if l[-3:] == 'jpg']
    print("Generate {} {} files from {} raw data...".format(data_size, new_path, len(filenames)))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for i in range(data_size):
        image_path = random.choice(filenames)
        np.save(new_path + '/' + ('%s' % i).zfill(6), ImagePreProcessing(image_path, rho, patch_size, imsize))
        if (i + 1) % 1000 == 0:
            print('--image number ', i+1)


if __name__ == "__main__":
    start = time.time()
    rho = 32
    patch_size = 128
    imsize = (320, 240)
    savedata(train_path, './training/', rho, patch_size, imsize, data_size=500000)
    savedata(val_path, './validation/', rho, patch_size, imsize, data_size=5000)
    savedata(test_path, './testing/', rho, patch_size, imsize, data_size=5000)
    elapsed_time = time.time() - start
    print("Generate dataset in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))

    # # show sample
    # from matplotlib import pyplot as plt
    # npy = random.choice([os.path.join('./training/', f) for f in os.listdir('./training/')])
    # ori_images, pts1, delta = np.load(npy, allow_pickle=True)
    # pts2 = pts1 + delta
    # patch1 = ori_images[:, :, 0].copy()
    # patch2 = ori_images[:, :, 1].copy()
    # patch1 = cv2.cvtColor(patch1, cv2.COLOR_GRAY2RGB)
    # patch2 = cv2.cvtColor(patch2, cv2.COLOR_GRAY2RGB)
    # cv2.polylines(patch1, [pts1], True, (81, 167, 249), 2, cv2.LINE_AA)
    # cv2.polylines(patch1, [pts2], True, (111, 191, 64), 2, cv2.LINE_AA)
    # cv2.polylines(patch2, [pts1], True, (111, 191, 64), 2, cv2.LINE_AA)
    # plt.subplot(121), plt.imshow(patch1)
    # plt.subplot(122), plt.imshow(patch2)
    # plt.show()