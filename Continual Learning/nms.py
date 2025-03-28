import sys
import os
import argparse

import numpy as np
from skimage import filters

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


parser = argparse.ArgumentParser(description='Generate predicted coordinates'
                                 ' from probability map of tumor patch'
                                 ' predictions, using non-maximal suppression')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the input probs_map numpy file')
parser.add_argument('coord_path', default=None, metavar='COORD_PATH',
                    type=str, help='Path to the output coordinates csv file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' the probability map was generated, default 6,'
                    ' i.e. inference stride = 64')
parser.add_argument('--radius', default=6, type=int, help='radius for nms,'
                    ' default 12 (6 used in Google paper at level 7,'
                    ' i.e. inference stride = 128)')
parser.add_argument('--prob_thred', default=0.5, type=float,
                    help='probability threshold for stopping, default 0.5')
parser.add_argument('--sigma', default=0.0, type=float,
                    help='sigma for Gaussian filter smoothing, default 0.0,'
                    ' which means disabled')


def run(args,probs_map_path,coord_path):
    probs_map = np.load(probs_map_path)
    X, Y = probs_map.shape
    #resolution = pow(2, args.level)
    resolution = pow(2, 3)

    if args.sigma > 0:
        probs_map = filters.gaussian(probs_map, sigma=args.sigma)

    outfile = open(coord_path, 'w')
    while np.max(probs_map) > args.prob_thred:
        prob_max = probs_map.max()
        max_idx = np.where(probs_map == prob_max)
        x_mask, y_mask = max_idx[0][0], max_idx[1][0]
        x_wsi = x_mask#int((x_mask + 0.5) * resolution)
        y_wsi = y_mask#int((y_mask + 0.5) * resolution)
        outfile.write('{:0.5f},{},{}'.format(prob_max, x_wsi, y_wsi) + '\n')

        x_min = x_mask - args.radius if x_mask - args.radius > 0 else 0
        x_max = x_mask + args.radius if x_mask + args.radius <= X else X
        y_min = y_mask - args.radius if y_mask - args.radius > 0 else 0
        y_max = y_mask + args.radius if y_mask + args.radius <= Y else Y

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                probs_map[x, y] = 0

    outfile.close()


def main():
    args = parser.parse_args()
    files = os.listdir(args.probs_map_path)
    done = os.listdir(args.coord_path)
    ignore = []
    #    '103','093','098','107','099','096', # prontos
    #    '110','109','108','100','102','097','104','105' # pesados
    #]
    
    for file in files:
        file = file.replace('.npy','')
        if (file+'.csv' in done) or (file in ignore):
            print(file + ' ignored')
            continue
        coord_path = args.coord_path + file+'.csv'
        probs_map_path = args.probs_map_path + file+'.npy'
        print(file + ' running')
        run(args,probs_map_path,coord_path)

    print('DONE')

if __name__ == '__main__':
    main()

# https://github.com/wollf2008/FW-RD