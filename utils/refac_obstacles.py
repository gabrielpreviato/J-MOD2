import numpy as np
import csv
import os
from glob import glob
import math

dataset_main_dir = '/home/previato/LaRoCS/dataset'
test_dir = 'data3'

obs_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'obstacles_10m', '*' + '.txt')))

def read_labels_gt_viewer(obstacles_gt):
    with open(obstacles_gt, 'r') as f:
        obstacles = f.readlines()
    obstacles = [x.strip() for x in obstacles]

    labels = []

    for obs in obstacles:
        parsed_str_obs = obs.split(" ")
        parsed_obs = np.zeros(shape=(8))
        i = 0
        for n in parsed_str_obs:
            if i < 2:
                parsed_obs[i] = int(n)
            else:
                parsed_obs[i] = float(n)
            i += 1

        object = [int(parsed_obs[0]*32 + parsed_obs[2]*32), int(parsed_obs[1]*32 + parsed_obs[3]*32),
                  int(parsed_obs[4]*256), int(parsed_obs[5]*160), parsed_obs[6], parsed_obs[7]]

        x, y, w, h, m, v = object

        new_object = [int(math.floor((x + w / 2) / 32)), int(math.floor((y + h / 2) / 32)),
                      math.floor((x + w / 2) % 32) / 32.0, math.floor((y + h / 2) % 32) / 32.0,
                      w / 256.0, h / 160.0, m, v
                     ]

        labels.append(new_object)

    with open(obstacles_gt, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=' ')

        for obs in labels:
            writer.writerow(obs)


for obs_path in obs_paths:
    print obs_path
    obs = read_labels_gt_viewer(obs_path)
