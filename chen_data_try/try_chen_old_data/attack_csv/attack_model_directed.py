import pandas as pd
import numpy as np
import csv
import random
from random import randint

import os


def launch_attack(csv_input):
    # number of data points in csv data file
    # num_rows = csv_input.shape[0]
    # num_cols = csv_input.shape[1]

    # # vec = [6, 12, 17]
    # # col =  vec[randint(0,2)]

    # col = 6
    # # store the headers of csv files in labels
    # label = list(csv_input)
    # seq = list(range(2, num_rows - 1))
    # print(seq)
    # a = random.sample(seq, 4081)
    # a = random.sample(seq, (int(0.3*num_rows)))
    # a.sort()
    # print("a: ", a)

    # for row in a:

    #     val = csv_input['angle_convert_org'].iloc[row]
    #     #     print (val)

    #     # Randomly select 30% images and for an image, add or subtract random value in [0.2, 0.9] to its corresponding angle
    #     rnd = random.uniform(0.2, 0.9)

    #     # add or subtract
    #     temp = randint(1, 2)
    #     if temp == 2:
    #         # add
    #         operation = 1
    #     else:
    #         #subtract
    #         operation = -1

    #     if val == 'NAN':
    #         return
    #     else:
    #         # csv_input.set_value(row, label[col], str(val + operation * rnd))
    #         # csv_input.set_value(row, label[num_cols - 1], '1')
    #         csv_input.at['angle_new',row] =str(val + operation * rnd)
    #         csv_input.at['Anomaly',row]=str(1)
    # Select the largest 15% and smallest 15% angles. Flip the sign of a selected angle if its absolute value is larger than 0.3.
    # https://stackoverflow.com/questions/46450260/how-can-i-randomly-change-the-values-of-some-rows-in-a-pandas-dataframe
    top15 = csv_input.nlargest(int(csv_input.shape[0] * 0.15), "angle_new")
    btm15 = csv_input.nsmallest(int(csv_input.shape[0] * 0.15), "angle_new")
    dfupdate = pd.concat([top15, btm15])
    dfupdate["Anomaly"] = "1"
    # If not, add or subtract a random value in [0.5, 1].
    dfupdate["random_rad"] = [random.uniform(0.5, 0.1) if value <= 0.3 else 0 for value in dfupdate["angle_new"]]
    dfupdate["add_subt"] = [random.choice([-1, 1]) if value <= 0.3 else 0 for value in dfupdate["angle_new"]]
    # Flip the sign of a selected angle if its absolute value is larger than 0.3.
    dfupdate["flip"] = [-1 if value > 0.3 else 1 for value in dfupdate["angle_new"]]

    dfupdate["angle_new"] = dfupdate["angle_new"] * dfupdate["flip"] + (dfupdate["add_subt"] * dfupdate["random_rad"])
    csv_input.update(dfupdate)
    return csv_input


if __name__ == "__main__":
    # # root_dir = "/home/ubuntu/try_HMB/data_HMB2/outp"
    # root_dir = "./chen/chen_old"

    # csv_file = "interpolated_center_10702to24371_part1.csv"
    # # print("root_dir: ", root_dir)

    # folders = os.listdir(root_dir)
    # all_folders = []

    # for folder in folders:
    #     if os.path.isdir(os.path.join(root_dir, folder)):
    #         all_folders.append(os.path.join(root_dir, folder))

    # # print(all_folders)

    # for folder in all_folders:
    data_path = "/home/ubuntu/RAIDS/dataset/chen_old/"
    csv_input = pd.read_csv("{}chen_old_all.csv".format(data_path))
    print(csv_input.tail(10))
    # print("csv_input: ", csv_input)
    # create new column Anomaly
    csv_input["Anomaly"] = "0"
    csv_input["random_rad"] = np.nan
    csv_input["add_subt"] = np.nan
    csv_input["flip"] = np.nan

    # copy original angle in rad to new angle for attacks later
    csv_input["angle_new"] = csv_input["angle_convert_org"]

    csv_input = launch_attack(csv_input)

    # print(csv_input)
    # print(type(folder))
    #  folder=''.join('/interpolated_attack.csv')
    # print(os.path.join(folder, 'interpolated_attack_ABS_center_old.csv'))
    # csv_input.to_csv(os.path.join(folder, 'interpolated_random_a_10702to24371_part1_0p2to0p9.csv'), index = False)
    print(csv_input.tail(10))
    csv_input.to_csv("chen_old_all_directed_intrusion.csv", index=False)
