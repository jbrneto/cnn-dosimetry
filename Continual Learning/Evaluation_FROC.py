# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016
@author: Babak Ehteshami Bejnordi
Evaluation code for the Camelyon16 challenge on cancer metastases detecion
"""

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import sys

def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float: 
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

def computeEvaluationMask(maskDIR, resolution, level):
    print(maskDIR, resolution, level)
    """Computes the evaluation mask.
    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made
    Returns:
        evaluation_mask
    """
    slide = openslide.open_slide(maskDIR)
    dims = slide.level_dimensions[level]

    #mask_image = read_slide(slide, x=0, y=0, level=level, width=dims[0], height=dims[1])
    
    pixelarray = np.zeros(dims[0] * dims[1], dtype='uint')
    pixelarray = np.array(slide.read_region((0, 0), level, dims))
    distance = nd.distance_transform_edt(255 - pixelarray[:, :, 0])
    Threshold = 75 / (resolution * pow(2, level) * 2)  # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity=2)
    
    #print('mask_image shape', mask_image.shape)
    print('evaluation_mask shape', evaluation_mask.shape)
    
    return evaluation_mask#, mask_image


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)
    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.
    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made
    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275 / (resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i + 1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR, "r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        Probs.append(float(elems[0]))
        Xcorr.append(int(elems[1]))
        Ycorr.append(int(elems[2]))
    return Probs, Xcorr, Ycorr


def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image
    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made
    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections
        TP_probs:   A list containing the probabilities of the True positive detections
        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)
        detection_summary:   A python dictionary object with keys that are the labels
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate].
        Lesions that are missed by the algorithm have an empty value.
        FP_summary:   A python dictionary object with keys that represent the
        false positive finding number and values that contain detection
        details [confidence score, X-coordinate, Y-coordinate].
    """

    max_label = np.amax(evaluation_mask)
    print('max_label', max_label)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1, max_label + 1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []

    TP_counter = 0
    TP_else_counter = 0
    FP_counter = 0
    if (is_tumor) or True:
        #print('Isolated_Tumor_Cells', Isolated_Tumor_Cells)
        for i in range(0, len(Xcorr)):
            #print(i, '=>', Ycorr[i], Xcorr[i], '=>', int(Ycorr[i] / pow(2, level)), int(Xcorr[i] / pow(2, level)))
            #HittedLabel = evaluation_mask[int(Ycorr[i] / pow(2, level)), int(Xcorr[i] / pow(2, level))]
            HittedLabel = evaluation_mask[int(Xcorr[i]),int(Ycorr[i])]
            #print(HittedLabel)
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter += 1
            elif HittedLabel not in Isolated_Tumor_Cells:
            #else:
                TP_else_counter += 1
                #print(Probs[i], '>', TP_probs[HittedLabel - 1])
                if (Probs[i] > TP_probs[HittedLabel - 1]):
                #if ((Probs[i] > 0) and (TP_probs[HittedLabel - 1] > 0)):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel - 1] = Probs[i]
                    TP_counter += 1
    else:
        for i in range(0, len(Xcorr)):
            FP_probs.append(Probs[i])
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
            FP_counter += 1

    num_of_tumors = max_label - len(Isolated_Tumor_Cells);
    print('Isolated_Tumor_Cells', Isolated_Tumor_Cells)
    print('num_of_tumors', num_of_tumors)
    #print(FP_probs,TP_probs)
    print('TP', TP_counter, 'TPelse', TP_else_counter, 'FP', FP_counter)
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary


def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs) / float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs) / float(sum(FROC_data[3]))
    
    return total_FPs, total_sensitivity


def plotFROC(total_FPs, total_sensitivity):
    """Plots the FROC curve
    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    Returns:
        -
    """
    plt.figure(0).clf()
    plt.plot(total_FPs, total_sensitivity, '-', color='r', label=r'Our = %0.3f' % float(sum(total_sensitivity)/6))
    plt.xlim([0.25, 8])
    plt.ylim([0, 1])
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)
    plt.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    lg = plt.legend(loc='lower right', borderaxespad=1.)
    lg.get_frame().set_edgecolor('k')
    plt.grid(True, linestyle='-')
    plt.show()
if __name__ == "__main__":
    mask_folder = sys.argv[1]
    result_folder = sys.argv[2]
    result_file_list = []
    ignored_cases = ['051.csv','030.csv','068.csv','002.csv','075.csv','074.csv','082.csv']
    #ignored_cases = []
    result_file_list += [each for each in os.listdir(result_folder) if (each.endswith('.csv') and (each not in ignored_cases))]
    #result_file_list = ['099.csv','107.csv','098.csv','093.csv','103.csv']#,'096.csv']
    #result_file_list = ['121.csv','061.csv','108.csv','097.csv','064.csv','038.csv','104.csv','027.csv']

    EVALUATION_MASK_LEVEL = 3  # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243  # pixel resolution at level 0
    #L0_RESOLUTION = 1.0

    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)

    ground_truth_test = []
    ground_truth_test += [each[0:8] for each in os.listdir(mask_folder) if each.endswith('.tif')]
    ground_truth_test = set(ground_truth_test)

    caseNum = 0
    for case in result_file_list:
        print('Evaluating Performance on image:', case[0:-4])
        sys.stdout.flush()
        csvDIR = os.path.join(result_folder, case)
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)

        is_tumor = True#case[0:-4] in ground_truth_test
        if (is_tumor):
            maskDIR =  os.path.join(mask_folder, 'tumor_'+case[0:-4]) + '_mask.tif'
            evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            #ITC_labels = []
        else:
            evaluation_mask = 0
            ITC_labels = []

        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], \
        FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels,
                                                     EVALUATION_MASK_LEVEL)
        caseNum += 1

    # Compute FROC curve
    total_FPs, total_sensitivity = computeFROC(FROC_data)

    # plot FROC curve
    plotFROC(total_FPs, total_sensitivity)
    print(total_FPs,total_sensitivity)

    eval_threshold = [.25, .5, 1, 2, 4, 8]
    eval_TPs = np.interp(eval_threshold, total_FPs[::-1], total_sensitivity[::-1])
    plotFROC(eval_threshold, eval_TPs)

    for i in range(len(eval_threshold)):
        print('Avg FP = ', str(eval_threshold[i]))
        print('Sensitivity = ', str(eval_TPs[i]))

    print('Avg Sensivity = ', np.mean(eval_TPs))
    print('DONE')
# https://github.com/wollf2008/FW-RD