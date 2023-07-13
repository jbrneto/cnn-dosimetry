import numpy as np

def calculate_3d_volume(object):
    area = 0.0
    for depth in range(object.shape[2]):
        layer = np.nonzero(object[:,:,depth])
        x, y = layer[0], layer[1]
        area = area + (abs(sum(x[i] * (y[i + 1] - y[i - 1]) for i in range(-1, len(x) - 1))) / 2.0)
    return area

def calc_conformity(ptv, prediction):
    target = (ptv > 0).astype(float)
    prescription = (prediction > 0).astype(float)
    intersection = target * prescription
    
    tv = calculate_3d_volume(target)
    pv = calculate_3d_volume(prescription)
    tvXpv = calculate_3d_volume(intersection)
    
    CI = (tvXpv*tvXpv)/(tv*pv)
    #print('TV', tv, 'PV', pv, 'TvXPv', tvXpv)
    #print(CI)
    return CI

def calc_DVH(prediction):
    step = 0.01
    
    dose_vals = prediction[np.where(prediction > 0)].ravel()
    bins = np.arange(0, 1.0 + step, step)
    
    hist_values = np.histogram(dose_vals, bins)[0]
    
    values = np.cumsum(hist_values[::-1])[::-1]
    values = values / values.max()

    return values, bins[:-1]

def calculate_dose_volume(dvh_values, dvh_bins, threshold):
    HI = np.interp(threshold / 100, dvh_values[::-1], dvh_bins[::-1])
    if dvh_values[0] == np.sum(dvh_values):
        HI = 0
    return HI

def calc_homogeneity(prediction):
    dvh_values, dvh_bins = calc_DVH(prediction)
    d2 = calculate_dose_volume(dvh_values, dvh_bins, 2)
    d50 = calculate_dose_volume(dvh_values, dvh_bins, 50)
    d98 = calculate_dose_volume(dvh_values, dvh_bins, 98)
    #print(d2, d98, d50)
    return (d2 - d98) / d50