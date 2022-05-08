# from __future__ import print_function, division
from sklearn.model_selection import train_test_split
import os
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
import torch.optim as optim
from data_extraction_ABS import *
from torch.autograd import Variable

from torch.optim import lr_scheduler
from torch.autograd import Variable
from model_abs_linear import *

import numpy as np
from keras.models import load_model
import pandas as pd
import glob
from config import TestConfig
from train import *
from time import gmtime, strftime

import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import datetime

time_a = datetime.datetime.now()
print("intrusion_detection_start: ", time_a)


class Model(object):
    def __init__(self, model_path, X_train_mean_path):

        self.model = model_path
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img_path):
        img1 = load_img(img_path, grayscale=True, target_size=(100, 100))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)  # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return self.mean_angle

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))

            img = np.array(img, dtype=np.uint8)  # to replicate initial model

            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)
            X = X[:, :, ::-1]

            X = np.expand_dims(X, axis=0)
            X = X.astype("float32")
            X -= self.X_mean
            X /= 255.0

            return self.model.predict(X)[0]


if __name__ == "__main__":
    # face_dataset = intrusion_data(csv_file='interpolated_center_attack_consecutive_0p2to0p9_all_train.csv',
    #                               root_dir='/home/pi/new_try_raspberry_pi/udacity_data_try_pi/all',
    #                               cs=1,transform=transforms.Compose([transforms.Resize(256),
    #                                                                                    transforms.RandomResizedCrop(224),transforms.ToTensor()]))
    # test_dataset=intrusion_data(csv_file='interpolated_center_attack_consecutive_0p2to0p9_all_test.csv',
    #                             root_dir='/home/pi/new_try_raspberry_pi/udacity_data_try_pi/all',
    #                             cs=1,transform=transforms.Compose([transforms.Resize(256),
    #                                                                                  transforms.RandomResizedCrop(224),transforms.ToTensor()]))
    # # out=[]
    # dataloader = DataLoader(face_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # dataset = pd.read_csv(
    #     "/home/ubuntu/RAIDS/udacity_data_try/try_udacity_data_2cnn/attack_csv/udacity_all_abrupt_intrusion.csv"
    # )
    dataset = pd.read_csv(
        "/home/ubuntu/RAIDS/udacity_data_try/try_udacity_data_2cnn/attack_csv/udacity_all_directed_intrusion.csv"
    )
    # split the dataset into training and test. needs to be in order cos the images are substracted
    face_dataset, test_dataset = train_test_split(dataset, test_size=0.3, shuffle=False, random_state=56)
    face_dataset = intrusion_data(face_dataset, root_dir="/home/ubuntu/RAIDS/dataset/udacity/CH2_002/output")
    test_dataset = intrusion_data(test_dataset, root_dir="/home/ubuntu/RAIDS/dataset/udacity/CH2_002/output")

    # It represents a Python iterable over a datase
    dataloader = DataLoader(face_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    config = TestConfig()

    ch = config.num_channels
    row = config.img_height
    col = config.img_width
    model_org = create_comma_model_large_dropout(
        row,
        col,
        ch,
        load_weights=True,
        path="/home/ubuntu/RAIDS/udacity_data_try/feature_extraction_100m100_2cnn/Model.h5",
    )

    model = Model(model_org, "data/X_train_gray_diff2_mean.npy")

    # Neural networks can be constructed using the torch.nn package.
    net = mynet()
    criterion = F.nll_loss  # The negative log likelihood loss.
    optimizer = optim.Adam(net.parameters(), lr=0.01)  # Implements Adam algorithm.

    net.eval()
    net = torch.load("saved_consecutive_models/epoch_39.pt")

    df_all = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])
    y_pred = []
    y_true = []

    print("Length of dataloader :", len(test_loader))
    inter = 0.01

    total_time = datetime.timedelta()
    max_time = datetime.timedelta()

    for i_batch, sample in enumerate(test_loader):
        if i_batch / len(test_loader) > inter:
            print("epoch: test", " completed: ", inter * 100, "%")
            inter += 0.01

        data = []
        result = []
        for sm in sample:
            imag, dat, res = sm
            data = dat
            result.append(res)

        for img in imag:
            time1 = datetime.datetime.now()

            preds = model.predict(img)
            preds = preds.astype("float").reshape(-1)
            preds = preds[0]

        target = torch.stack(result)
        target = target.view(-1)
        y_true.extend(target)

        final_vars = []
        final_vars = torch.FloatTensor([[[abs(data - preds)]]])

        # forward + backward + optimize
        x = net(*final_vars)
        values, indices = x.max(1)
        y_pred.extend(indices.data)

        time3 = datetime.datetime.now()
        # print("detection_result: ", time3)
        k = time3 - time1
        # print("input-output: ", k)
        # k = int(k)
        if max_time < k:
            max_time = k
        total_time = total_time + k

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred).flatten()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print("TN, FP, FN, TP :", cf_matrix)
    print("accuracy :", accuracy)

    # Initialize a blank dataframe and keep adding
    df = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])
    print(type(cf_matrix))
    df.loc["time"] = cf_matrix.tolist() + [accuracy, precision, recall]
    df["Total_Actual_Neg"] = df["TN"] + df["FP"]
    df["Total_Actual_Pos"] = df["FN"] + df["TP"]
    df["Total_Pred_Neg"] = df["TN"] + df["FN"]
    df["Total_Pred_Pos"] = df["FP"] + df["TP"]
    df["TP_Rate"] = df["TP"] / df["Total_Actual_Pos"]  # Recall
    df["FP_Rate"] = df["FP"] / df["Total_Actual_Neg"]
    df["TN_Rate"] = df["TN"] / df["Total_Actual_Neg"]
    df["FN_Rate"] = df["FN"] / df["Total_Actual_Pos"]
    df["max_time"] = max_time
    df["total_time"] = total_time
    df["avg_time"] = total_time / len(test_loader)

    # df.to_csv("accuracy_file_f_e_predict_abrupt_time.csv")
    df.to_csv("accuracy_file_f_e_predict_directed_time.csv")

    time_b = datetime.datetime.now()
    print("intrusion_detection_end: ", time_b)
    print("all_time: ", time_b - time_a)
