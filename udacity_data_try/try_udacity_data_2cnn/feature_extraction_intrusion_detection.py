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

import numpy as np
from keras.models import load_model
import pandas as pd
import glob
from config import TestConfig
from train import *

import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


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

    net = mynet()

    criterion = F.nll_loss
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    epochs = 40
    cnt = 0
    df_all = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])

    for iter_x in range(epochs):
        inter = 0.01
        loss1 = 0
        finale = 0
        # train_loss = 0
        finale_1 = 0
        finale_0 = 0
        get_1 = 0
        get_0 = 0
        target_1_indices_1 = 0
        target_1_indices_0 = 0
        target_0_indices_1 = 0
        target_0_indices_0 = 0

        print("Length of dataloader :", len(dataloader))
        for i_batch, sample in enumerate(dataloader):  # for each training i_batch

            if i_batch / len(dataloader) > inter:
                print("epoch: ", iter_x, " completed: ", inter * 100, "%")
                inter += 0.01

            data = []
            result = []
            for sm in sample:
                imag, dat, res = sm
                data = dat
                result.append(res)

            target = torch.stack(result)
            target = target.view(-1)

            # prediction from stage 1 model
            for img in imag:
                preds = model.predict(img)
                preds = preds.astype("float").reshape(-1)
                preds = preds[0]

            final_vars = []
            final_vars = torch.FloatTensor([[[abs(data - preds)]]])

            # forward + backward + optimize
            x = net(*final_vars)
            values, indices = x.max(1)
            loss = criterion(x, Variable(target.long()))  # The negative log likelihood loss.
            loss.backward()
            optimizer.step()
            # zero the parameter gradients
            net.zero_grad()

            # if(i_batch>10):
            #     break

        print("TESTING")
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&TESTING &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        y_pred = []
        y_true = []

        for i_batch, sample in enumerate(test_loader):

            data = []
            result = []
            for sm in sample:
                imag, dat, res = sm
                data = dat
                result.append(res)

            for img in imag:
                preds = model.predict(img)
                preds = preds.astype("float").reshape(-1)
                preds = preds[0]

            final_vars = torch.FloatTensor([[[abs(data - preds)]]])
            target = torch.stack(result)
            target = target.view(-1)
            y_true.extend(target)

            x = net(*final_vars)
            values, indices = x.max(1)  # takes the max over dimension 1 and returns two values.
            # print("target_indices: ", target, indices.data)
            y_pred.extend(indices.data)

            # # number of wrong prediction
            # loss1 += myaccuracy(target, indices.data)
            # # number of test cases
            # finale += 1

            # for Target, Indices in zip(target, indices.data):
            #     if Target == 1: # number of test cases with attack
            #         finale_1 += 1
            #     else:
            #         finale_0 += 1
            #     if Indices == 1: # number of test cases predicted under attack
            #         get_1 += 1
            #     else:
            #         get_0 += 1
            #     if Target == 1 and Indices == 1: # True positive
            #         target_1_indices_1 += 1
            #     elif Target == 1 and Indices == 0: # False Negative
            #         target_1_indices_0 += 1
            #     elif Target == 0 and Indices == 0:  # True negative
            #         target_0_indices_0 += 1
            #     elif Target == 0 and Indices == 1: # False positive
            #         target_0_indices_1 += 1

            # if(i_batch>10):
            #     break
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
        df.loc[iter_x] = cf_matrix.tolist() + [accuracy, precision, recall]
        df["Total_Actual_Neg"] = df["TN"] + df["FP"]
        df["Total_Actual_Pos"] = df["FN"] + df["TP"]
        df["Total_Pred_Neg"] = df["TN"] + df["FN"]
        df["Total_Pred_Pos"] = df["FP"] + df["TP"]
        df["TP_Rate"] = df["TP"] / df["Total_Actual_Pos"]  # Recall
        df["FP_Rate"] = df["FP"] / df["Total_Actual_Neg"]
        df["TN_Rate"] = df["TN"] / df["Total_Actual_Neg"]
        df["FN_Rate"] = df["FN"] / df["Total_Actual_Pos"]
        df_all = pd.concat([df_all, df])
        print(df_all.tail())

        # # confusion matrix for a binary classifier
        # acc = 1.0 - (loss1 / finale)
        # acc11 = target_1_indices_1 / (target_1_indices_1 + target_1_indices_0)
        # print('Accuracy=', acc) # overall accuracy (TP+TN)/total
        # print('True Positive Rate=', acc11) # True Positive Rate TP/actual yes
        # print("finale_1_0_get_1_0: ", finale_1, finale_0, get_1, get_0)  # Actual=Yes, Actual=No, Predicted = Yes, Predicted = No
        # # True Positive, False Negative, True Negative, False Positive
        # print("target_indices11,10,00,01: ", target_1_indices_1, target_1_indices_0, target_0_indices_0, target_0_indices_1)

        # # Initialize a blank dataframe and keep adding
        # file2write = open("accuracy_file_f_e_predict_a_consecutive.txt", 'a')
        # file2write.write("accuracy= " + str(acc) + "\n")
        # file2write.write("accuracy1111= " + str(acc11) + "\n")

        # file2write.write(
        #     "finale_1_0_get_1_0= " + "\n" + str(finale_1) + "\n" + str(finale_0) + "\n" + str(get_1) + "\n" + str(
        #         get_0) + "\n")
        # file2write.write(
        #     "target_indices11,10,00,01= " + "\n" + str(target_1_indices_1) + "\n" + str(target_1_indices_0) + "\n" + str(
        #         target_0_indices_0) + "\n" + str(target_0_indices_1) + "\n")
        # file2write.close()

        filenm = "epoch_" + str(cnt) + ".pt"
        torch.save(net, os.path.join(("saved_consecutive_models"), filenm))

        cnt = cnt + 1
        if cnt % 10 == 0:
            print("EPOCH completed by %", (cnt / 40) * 100)

    # df_all.to_csv("accuracy_file_f_e_predict_abrupt.csv")
    df_all.to_csv("accuracy_file_f_e_predict_directed.csv")
