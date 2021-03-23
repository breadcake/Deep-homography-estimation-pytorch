import torch
from torch.utils.data import DataLoader
from dataset import CocoDdataset
from model import HomographyNet
import argparse
import os
import numpy as np
import cv2
import utils


def denorm_img(img):
    img = img * 127.5 + 127.5
    img = np.clip(img, 0, 255)
    return np.uint8(img)


def warp_pts(H, src_pts):
    src_homo = np.hstack((src_pts, np.ones((4, 1)))).T
    dst_pts = np.matmul(H, src_homo)
    dst_pts = dst_pts / dst_pts[-1]
    return dst_pts.T[:, :2]


def test(args):
    MODEL_SAVE_DIR = 'checkpoints/'
    model_path = os.path.join(MODEL_SAVE_DIR, args.checkpoint)
    result_dir = 'results/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    model = HomographyNet()
    state = torch.load(model_path)
    model.load_state_dict(state['state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()

    TestingData = CocoDdataset(args.test_path)
    test_loader = DataLoader(TestingData, batch_size=1)

    print("start testing")
    with torch.no_grad():
        model.eval()
        error = np.zeros(len(TestingData))
        for i, batch_value in enumerate(test_loader):
            ori_images = batch_value[0].float()
            inputs = batch_value[1].float()
            pts1 = batch_value[2]
            target = batch_value[3].float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            outputs = model(inputs)
            outputs = outputs * 32
            target = target * 32

            # visual
            I_A = denorm_img(ori_images[0, 0, ...].numpy())
            I_B = denorm_img(ori_images[0, 1, ...].numpy())
            pts1 = pts1[0].numpy()

            gt_h4p = target[0].numpy()
            pts2 = pts1 + gt_h4p
            gt_h = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
            gt_h_inv = np.linalg.inv(gt_h)
            pts1_ = warp_pts(gt_h_inv, pts1)

            pred_h4p = outputs[0].cpu().numpy().reshape([4, 2])
            pred_pts2 = pts1 + pred_h4p
            pred_h = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pred_pts2))
            pred_h_inv = np.linalg.inv(pred_h)
            pred_pts1_ = warp_pts(pred_h_inv, pts1)

            visual_file_name = ('%s' % i).zfill(4) + '.jpg'
            utils.save_correspondences_img(I_A, I_B, pts1, pts1_, pred_pts1_,
                                           result_dir, visual_file_name)

            error[i] = np.mean(np.sqrt(np.sum((gt_h4p - pred_h4p) ** 2, axis=-1)))
            print('Mean Corner Error: ', error[i])

        print('Mean Average Corner Error over the test set: ', np.mean(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="homographymodel.pth")
    parser.add_argument("--test_path", type=str, default="data/testing/", help="path to test images")
    args = parser.parse_args()
    test(args)
