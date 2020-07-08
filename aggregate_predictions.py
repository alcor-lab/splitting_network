import os
import numpy as np
from tqdm import tqdm
import itertools
from itertools import islice,repeat
from functools import partial
import copy
import pickle 
from joblib import Parallel, delayed
from time import sleep
from contextlib import contextmanager
import argparse
from torch.utils.data import Dataset, DataLoader
from videoDataset import videoDataset
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Videos to images')
parser.add_argument('--arch', type=str,default='I3D', help='architecture type. options: I3D, Attention, P3D')
parser.add_argument('--data_dir', type=str, help="data dir", default = "/home/paolo/softlinkFolder/")
parser.add_argument('--train_file', type=str, help="train file list", default = None)
parser.add_argument('--test_file', type=str, help="test file list", default = None)
parser.add_argument('--test',action="store_true")
parser.add_argument('--fix_imbalance',action="store_true", help="normalize training class frequency")
parser.add_argument('--test_batch_size',type=int, help="test batch size", default=None)
parser.add_argument('--class_list',type=str, help="class list for the 20 classes category", default="./config/class_list_fixed.txt")
parser.add_argument('--seed', help="choose seed", default=None)
parser.add_argument('--start_from',type=int, help="from which split the training should start; default 0", default=0)
parser.add_argument('--end_to',type=int, help="from which split the training should start; default 0", default=34)
parser.add_argument('--num_classes',type=int, help="number of classes", default=2)

args = parser.parse_args()


print("name of the current run: "+str(args.seed))
print("//////////////")

def aggregate_pred(args):
    data_dir = args.data_dir
    video_dataset = videoDataset(data_dir,channels=3,timeDepth=300,size=224,crop_time=1,xCrop=None,yCrop=None,sequence_first=False,
                                 file_list=None,class_list=args.class_list,noVideo=True,multiclass=True)
    dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False)

    mat_list = []
    for i in range(args.start_from,args.end_to,1):
        mat = np.load("./results/"+str(args.seed)+"/pred_"+ str(args.seed) +"_"+ str(i)+".npy")
        mat_list.append(mat)
    filehandler = open("matricione.pkl","wb")
    pickle.dump(mat_list,filehandler)
    filehandler.close()
    Labels = []
    Labels2 = []
    mats = np.stack(mat_list)
    mats = np.moveaxis(mats,[0,1,2],[1,0,2])
    i = 0
    total = 0.0
    correct = 0.0
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    confusion_matrix2 = np.zeros((700, args.num_classes))
    for _,labels,labels2,_ in tqdm(dataloader):
        Labels.append(labels)
        Labels2.append(labels2)
        mat = mats[i,...] #load the i_th matrix relative to the i_th sample
        if args.type == "MEAN": ##taking mean of the predictions
            output = np.mean(mat,axis=0)
        if args.type == "MAX":  ##taking max of predictions, it is equal to trust into prediction of the most confident model
            output = np.max(mat,axis=0)
        if args.type == "MEAN OF THE BEST FIVE":  ##taking max of predictions, it is equal to trust into prediction of the most confident model
            temp_max = np.max(mat,axis=1)
            idx = (-temp_max).argsort()[:5] #indexes of the 5 most confident predictions
            mat = mat[idx] # now in mat I have only the 5 most confident predictions
            output = np.mean(mat,axis=0) #doing finally the mean of those predictions

        predicted = np.argmax(output)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
        i+=1

        for t, p in zip(labels.numpy().ravel(), predicted.ravel()):
            confusion_matrix[t.astype(int), p.astype(int)] += 1

        for t, p in zip(labels2.numpy().ravel(), predicted.ravel()):
            confusion_matrix2[t.astype(int), p.astype(int)] += 1

    accuracy = (correct / total) * 100.0
    print("Calculations for aggregation type: "+args.type)
    print('Accuracy = {}'.format( accuracy))
    print("Per class accuracy:")
    print(np.diag(confusion_matrix)/confusion_matrix.sum(1))
    print("///////////////",flush=True)
    df_cm = pd.DataFrame(confusion_matrix2, range(700), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    filehandler = open("labels.pkl","wb")
    pickle.dump(Labels,filehandler)
    filehandler.close()
    filehandler = open("labels2.pkl","wb")
    pickle.dump(Labels2,filehandler)
    filehandler.close()

    return 0


if __name__ == "__main__":
    args.type = "MEAN"
    aggregate_pred(args)
    args.type = "MAX"
    aggregate_pred(args)
    args.type = "MEAN OF THE BEST FIVE"
    aggregate_pred(args)
