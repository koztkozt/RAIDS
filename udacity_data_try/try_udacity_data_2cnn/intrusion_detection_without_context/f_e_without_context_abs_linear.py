from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data_extraction_ABS import *
from time import gmtime, strftime

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


if __name__ == "__main__":

    # face_dataset = intrusion_data(csv_file='interpolated_center_attack_consecutive_0p2to0p9_all_train.csv', root_dir='./csv_data', cs=1,
    #                             transform=transforms.Compose(
    #                                 [transforms.Resize(256),
    #                                 transforms.RandomResizedCrop(224),
    #                                 transforms.ToTensor()]))

    # test_dataset = intrusion_data(csv_file='interpolated_center_attack_consecutive_0p2to0p9_all_test.csv', root_dir='./csv_data', cs=1,
    #                             transform=transforms.Compose(
    #                                 [transforms.Resize(256),
    #                                 transforms.RandomResizedCrop(224),
    #                                 transforms.ToTensor()]))
    # out=[]

    # importing the dataset csv with attacks
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

    # out=[]

    # It represents a Python iterable over a datase
    dataloader = DataLoader(face_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Neural networks can be constructed using the torch.nn package.
    d_batch = 4
    net = mynet()
    # if use_gpu:
    #     net.cuda()
    criterion = F.nll_loss
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    epochs = 40
    cnt = 0
    df_all = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])

    for iter_x in range(epochs):
        inter = 0.01
        loss1 = 0
        finale = 0
        train_loss = 0
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
                data.append(dat.type(torch.FloatTensor))
                result.append(res)

            target = torch.stack(result)
            target = target.view(-1)

            final_vars = []
            for tens in data:
                final_vars.append(Variable(tens))

            # forward + backward + optimize
            x = net(*final_vars)
            values, indices = x.max(1)
            loss = criterion(x, Variable(target))  # The negative log likelihood loss.
            # train_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            net.zero_grad()

            # if i_batch > 10:
            #     break

        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&TESTING &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        print("test_start: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        y_pred = []
        y_true = []

        for i_batch, sample in enumerate(test_loader):

            data = []
            result = []

            for sm in sample:
                imag, dat, res = sm
                data.append(dat.type(torch.FloatTensor))
                result.append(res)

            final_vars = []
            for tens in data:
                final_vars.append(Variable(tens))

            target = torch.stack(result)
            target = target.view(-1)
            y_true.extend(target)

            x = net(*final_vars)
            values, indices = x.max(1)
            # print("target_indices: ", target, indices.data)
            y_pred.extend(indices.data)

        #     loss1 += accuracy(target, indices.data)
        #     finale += 4

        #     for Target, Indices in zip(target, indices.data):
        #         if Target == 1:
        #             finale_1 += 1
        #         else:
        #             finale_0 += 1
        #         if Indices == 1:
        #             get_1 += 1
        #         else:
        #             get_0 += 1
        #         if Target == 1 and Indices == 1:
        #             target_1_indices_1 += 1
        #         elif Target == 1 and Indices == 0:
        #             target_1_indices_0 += 1
        #         elif Target == 0 and Indices == 0:
        #             target_0_indices_0 += 1
        #         elif Target == 0 and Indices == 1:
        #             target_0_indices_1 += 1

        #     if i_batch > 10:
        #         break

        # acc = 1.0 - (loss1 / finale)
        # acc11 = target_1_indices_1 / (target_1_indices_1 + target_1_indices_0)
        # print("accuracy=", acc)
        # print("accuracy11=", acc11)
        # print("finale_1_0_get_1_0: ", finale_1, finale_0, get_1, get_0)
        # print(
        #     "target_indices11,10,00,01: ",
        #     target_1_indices_1,
        #     target_1_indices_0,
        #     target_0_indices_0,
        #     target_0_indices_1,
        # )

        # file2write = open("accuracy_file_f_e_without_context_abs_linear.txt", "a")
        # file2write.write("accuracy= " + str(acc) + "\n")
        # file2write.write("accuracy1111= " + str(acc11) + "\n")

        # file2write.write(
        #     "finale_1_0_get_1_0= "
        #     + "\n"
        #     + str(finale_1)
        #     + "\n"
        #     + str(finale_0)
        #     + "\n"
        #     + str(get_1)
        #     + "\n"
        #     + str(get_0)
        #     + "\n"
        # )
        # file2write.write(
        #     "target_indices11,10,00,01= "
        #     + "\n"
        #     + str(target_1_indices_1)
        #     + "\n"
        #     + str(target_1_indices_0)
        #     + "\n"
        #     + str(target_0_indices_0)
        #     + "\n"
        #     + str(target_0_indices_1)
        #     + "\n"
        # )
        # file2write.close()

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

        cnt = cnt + 1
        if cnt % 10 == 0:
            print("EPOCH completed by %", (cnt / 40) * 100)

        print("test_end: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # df_all.to_csv("accuracy_file_f_e_predict_a_abrupt.csv")
    df_all.to_csv("accuracy_file_f_e_predict_a_directed.csv")
