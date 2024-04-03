
import sys
sys.path.insert(0, '../')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import matplotlib.patches as mpatches
from unet.model import GeneralUNet
from utils.data_utils import BratsDataset3D
from utils.predict import ModelPredict
import test as gg
import SimpleITK as hh 
import seg_metrics.seg_metrics as sg
import testIoU_1 as tIoU
import warnings
import cv2
import scipy.ndimage
import skimage.filters
import sklearn.metrics

warnings.simplefilter('ignore')



# - dice:         Dice (F-1)
# - jaccard:      Jaccard or IoU
# - precision:    Precision
# - recall:       Recall
# - fpr:          False positive rate
# - fnr:          False negtive rate
# - vs:           Volume similarity

# - hd:           Hausdorff distance
# - hd95:         Hausdorff distance 95% percentile
# - msd:          Mean (Average) surface distance
# - mdsd:         Median surface distance
# - stdsd:        Std surface distance


class Eval_mri:
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def evaluation_emtrics(self, gdth_img, pred_img):
        labels = [0,1]
        metrics = sg.write_metrics(labels=labels[0:],  # exclude background if needed
                  gdth_img=gdth_img,
                  pred_img=pred_img)
        return metrics

    def generate_matrixConfusion(self, gdth_img, pred_img):
        gt_image = np.array(gdth_img, dtype=np.int64)
        pre_image = np.array(pred_img, dtype=np.int64)
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def get_confusion_matrix_intersection_mats(self,groundtruth, predicted):
        """
        Returns a dictionary of 4 boolean numpy arrays containing True at TP, FP, FN, TN.
        """
        confusion_matrix_arrs = {}

        groundtruth_inverse = np.logical_not(groundtruth)
        predicted_inverse = np.logical_not(predicted)

        confusion_matrix_arrs["tp"] = np.logical_and(groundtruth, predicted)
        confusion_matrix_arrs["tn"] = np.logical_and(groundtruth_inverse, predicted_inverse)
        confusion_matrix_arrs["fp"] = np.logical_and(groundtruth_inverse, predicted)
        confusion_matrix_arrs["fn"] = np.logical_and(groundtruth, predicted_inverse)

        return confusion_matrix_arrs

    def get_confusion_matrix_overlaid_mask(self,image, groundtruth, predicted, alpha, colors):
        """
        Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
        each a color given by the 'colors' dictionary
        """
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        masks = self.get_confusion_matrix_intersection_mats(groundtruth, predicted)
        color_mask = np.zeros_like(image)

        for label, mask in masks.items():
            color = colors[label]
            mask_rgb = np.zeros_like(image)
            mask_rgb[mask != 0] = color
            color_mask += mask_rgb

        return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)
