import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from videoDataset import videoDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
cudnn.benchmark = True
from tqdm import tqdm
from imbalanced import ImbalancedDatasetSampler
from models.objectAttentionModelConvLSTM import attentionModel
from models.p3d_model import P3D199
from models.pytorch_i3d import *
from models.ResNet2plus import R2Plus1DClassifier
from models.monster import frankestein
from models.ResNet3D import resnet3D34
import itertools
from itertools import islice,repeat
from functools import partial
import copy
from train_test_func import train, test
import pickle 
from joblib import Parallel, delayed
from time import sleep
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import gc
import multiprocessing
from contextlib import contextmanager
from pathlib import Path
import datetime

class ContinueI(Exception):
    pass

import argparse

parser = argparse.ArgumentParser(description='Videos to images')
parser.add_argument('--arch', type=str,default='I3D', help='architecture type. options: I3D, Attention, P3D')
parser.add_argument('--data_dir', type=str, help="data dir", default = "/home/paolo/softlinkFolder20vs20/")
parser.add_argument('--train_file', type=str, help="train file list", default = None)
parser.add_argument('--test_file', type=str, help="test file list", default = None)
parser.add_argument('--test',action="store_true")
parser.add_argument('--fix_imbalance',action="store_true", help="normalize training class frequency")
parser.add_argument('--no_parallel',action="store_true", help="normalize training class frequency")
parser.add_argument('--test_batch_size',type=int, help="test batch size", default=None)
parser.add_argument('--class_list',type=str, help="class list for the 20 classes category", default="./config/class_list_fixed.txt")
parser.add_argument('--seed', help="choose seed", default=None)
parser.add_argument('--start_from',type=int, help="from which split the training should start; default 0", default=0)
parser.add_argument('--gpu',type=int, help="test batch size", default=0)
parser.add_argument('--absolute_path',action="store_true", help="normalize training class frequency")
parser.add_argument('--rand_time',action="store_true", help="random ordered frames taken during training")
parser.add_argument('--epochs',type=int, help="number of epochs", default=None)
parser.add_argument('--early_stop',type=int, help="early_stop", default=None)

args = parser.parse_args()


GPU = 0
dropout = 0.0
args.num_classes = 2

def get_model(args):
    if args.arch == "I3D":
        model = InceptionI3d(num_classes=400, in_channels=3)
        model.load_state_dict(torch.load('models/i3d_rgb_imagenet.pt'))
        model.replace_logits(args.num_classes)
        args.sequence_first = False
        args.base_size = None
        args.xCrop = 224
        args.yCrop = 224
        args.crop_time = 64
        args.BATCH_SIZE = 4
        args.lr = 0.001
        return model,args
    elif args.arch == "I3D_Attention":
        model = InceptionI3d_Attention(num_classes=400, in_channels=3)
        model.load_state_dict(torch.load('models/i3d_rgb_imagenet.pt'))
        model.replace_logits(args.num_classes)
        args.sequence_first = False
        args.base_size = None
        args.xCrop = 224
        args.yCrop = 224
        args.crop_time = 64
        args.BATCH_SIZE = 4
        args.lr = 0.001
        return model,args
    elif args.arch == "I3D_Attention2":
        model = InceptionI3d_Attention2(num_classes=400, in_channels=3)
        model.load_state_dict(torch.load('models/i3d_rgb_imagenet.pt'))
        model.replace_logits(args.num_classes)
        args.sequence_first = False
        args.base_size = None
        args.xCrop = 224
        args.yCrop = 224
        args.crop_time = 64
        args.BATCH_SIZE = 4
        args.lr = 0.001
        return model,args
    elif args.arch == "ResNet":
        model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=True)
        model.fc = nn.Linear(512,2,bias=True)
        args.sequence_first = False
        args.base_size = None
        args.xCrop = 224
        args.yCrop = 224
        args.crop_time = 32
        args.BATCH_SIZE = 1
        args.lr = 0.00025
        if args.epochs is None:
            args.epochs = 4
        return model,args

if args.seed is None: 
    args.seed = np.random.randint(1000, size=1).item()

print("name of the current run: "+str(args.seed))
print("//////////////")
print("Current args:")
print(args)
nvmlInit() #to initialize nvml

def MyRun(i,args):
    gpu = args.gpu if args.no_parallel else i % 4 #gpu can be 0,1,2,3

    h = nvmlDeviceGetHandleByIndex(gpu)
    info = nvmlDeviceGetMemoryInfo(h)
#    while (info.used / (1024 * 1024)) > 2000.0:
#        print("Going to start, waiting for free memory in gpu n. "+str(gpu),flush=True)
#        gc.collect()
#        torch.cuda.empty_cache()
#        sleep(10)

    device = torch.device("cuda:"+str(gpu))
    model,args = get_model(args) #creating a new model
    model.to(device)
    a = datetime.datetime.now()
    Path("./results/" + str(args.seed)).mkdir(parents=True, exist_ok=True)
    if args.test and args.arch == "ResNet":
        data_dir = args.data_dir 
        model_path = "./results/" + str(args.seed)+ "/state_dict_"+ str(args.seed) +"_"+ str(i)
        state_dict = torch.load(model_path,map_location=device)
        model.load_state_dict(state_dict())
        model.eval()
        shuffle = False
        args.base_size = None
        args.xCrop = 224
        args.yCrop = 224
        cCrop = True
        args.BATCH_SIZE=8
    elif args.test:
        data_dir = args.data_dir 
        model_path = "./results/" + str(args.seed)+ "/state_dict_"+ str(args.seed) +"_"+ str(i)
        state_dict = torch.load(model_path,map_location=device)
        model.load_state_dict(state_dict())
        model.eval()
        shuffle = False
        args.base_size = None
        args.xCrop = 224
        args.yCrop = 224
        cCrop = True
        args.BATCH_SIZE=32
    else:
        data_dir = args.data_dir if args.absolute_path else args.data_dir + "/20_20_classes_"+str(i) + "/"
        shuffle = True
        cCrop = False

    b = datetime.datetime.now()
    delta = b - a
    print( str( int(delta.total_seconds() * 1000) ) )

    video_dataset = videoDataset(data_dir,channels=3,timeDepth=300,size=args.base_size,crop_time=args.crop_time,xCrop=args.xCrop,yCrop=args.yCrop,centerCrop=cCrop,
                                       sequence_first=args.sequence_first,file_list=args.train_file,class_list=args.class_list,rand_time=args.rand_time)
    dataloader = DataLoader(video_dataset, batch_size=args.BATCH_SIZE, num_workers=4, shuffle=shuffle)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)

    a = datetime.datetime.now()

    if args.test:
        pred = test(model,dataloader,args,device)
        np.save("./results/" + str(args.seed)+ "/pred_"+ str(args.seed) +"_"+ str(i),pred)
        print("saved predictions for gpu "+str(i))
    else:
        for epoch in range(args.epochs):
            train(model,dataloader,optimizer,args,device,epoch=epoch,early_stop=args.early_stop,n_thread=i)
            torch.save(model.state_dict,"./results/" + str(args.seed)+ "/state_dict_" + str(args.seed) +"_"+ str(i))
            print("saved training for gpu "+str(i))
            torch.cuda.empty_cache()
    b = datetime.datetime.now()
    delta = b - a
    print( str( int(delta.total_seconds() * 1000) ) )
    return 0


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

if __name__ == "__main__":
    #indexes = [i for i in range(args.start_from,34,1)]
    #print(indexes)

    #pool = multiprocessing.Pool(processes=4)
    #pool.map(Run,indexes)
    #pool.close()
#    with multiprocessing.Pool(processes=4) as pool:
#        pool.starmap(run,itertools.product(range(args.start_from,34,1), args))
#        pool.map(Run,args=range(args.start_from,34,1))
    if args.no_parallel:
        MyRun(args.start_from,args)
    else:
        for j in range(args.start_from,34,4):
            max_iter = j+4 if (j+4 <= 34) else 34
            Parallel(n_jobs=4,prefer='threads')(delayed(MyRun)(i,args) for i in range(j,max_iter,1))
            gc.collect()
            torch.cuda.empty_cache()
#Parallel(n_jobs=4,prefer='threads')(delayed(MyRun)(i,args) for i in range(args.start_from,34,1))

