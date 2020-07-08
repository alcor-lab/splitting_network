import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import islice
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.backends import cudnn
cudnn.benchmark = True


def train(model,data_loader,optimizer,args,device,epoch,early_stop=None,n_thread=None):
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.set_grad_enabled(True):
        model.train()
        running_loss = 0.0
        i = 0
        if early_stop is None:
            early_stop = len(data_loader)
        for videos,labels,files in islice(data_loader,0,early_stop):
#        for videos,labels,files in tqdm(data_loader):
            optimizer.zero_grad()
            i = i+1
            if args.sequence_first:
                videos = videos.permute(2,0,1,3,4)
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            if args.arch == "I3D":
                loss = F.binary_cross_entropy_with_logits(torch.max(outputs, dim=2)[0],F.one_hot(labels,num_classes=args.num_classes).float())
            else:
                loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if ((i+1) % 200) == 0:    # print every 200 mini-batches
                if n_thread is None:
                    print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss/200),flush=True)
                else:
                    print('Run ' + str(n_thread)+':[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss/200),flush=True)
                running_loss = 0.0
    return None

def test(model,data_loader,args,device,early_stop=None):
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    correct = 0.0
    OneCorr = 0.0
    ZeroCorr = 0.0
    i = 0
    total = 0.0
    confusion_matrix = torch.zeros(args.num_classes, args.num_classes)
    outputs_total = []

    if early_stop is None:
        early_stop = len(data_loader)
    with torch.no_grad():
        if early_stop is None:
            early_stop = len(data_loader)
#        for videos,labels,files in tqdm(islice(data_loader,0,early_stop)):
        for videos,labels,files in tqdm(data_loader):
            i = i+1
            if args.sequence_first:
                videos = videos.permute(2,0,1,3,4)
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            if args.arch == "I3D":
                loss = F.binary_cross_entropy_with_logits(torch.max(outputs, dim=2)[0],torch.nn.functional.one_hot(labels,num_classes=args.num_classes).float())
                outputs = torch.mean(outputs, 2)
            else:
                loss = criterion(outputs,labels)
            outputs_total.append(outputs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        accuracy = (correct / total) * 100.0
        print("///////////////")
        print('Accuracy = {}'.format( accuracy))
        print("Per class accuracy:")
        print(confusion_matrix.diag()/confusion_matrix.sum(1))
        print("///////////////",flush=True)
    return np.vstack(outputs_total)

def train_multiclass(model,data_loader,optimizer,args,device,epoch,early_stop=None,n_thread=None):
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.set_grad_enabled(True):
        model.train()
        running_loss = 0.0
        i = 0
        if early_stop is None:
            early_stop = len(data_loader)
        for videos,labels,files in islice(data_loader,0,early_stop):
#        for videos,labels,files in tqdm(data_loader):
            optimizer.zero_grad()
            i = i+1
            if args.sequence_first:
                videos = videos.permute(2,0,1,3,4)
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            if args.arch == "I3D":
                loss = F.cross_entropy(torch.max(outputs, dim=2)[0],labels)
            else:
                loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if ((i+1) % 200) == 0:    # print every 200 mini-batches
                if n_thread is None:
                    print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss/200),flush=True)
                else:
                    print('Run ' + str(n_thread)+':[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss/200),flush=True)
                running_loss = 0.0
    return None

def test_multiclass(model,data_loader,args,device,early_stop=None):
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    correct = 0.0
    OneCorr = 0.0
    ZeroCorr = 0.0
    i = 0
    total = 0.0
    confusion_matrix = torch.zeros(args.num_classes, args.num_classes)
    outputs_total = []

    if early_stop is None:
        early_stop = len(data_loader)
    with torch.no_grad():
        if early_stop is None:
            early_stop = len(data_loader)
#        for videos,labels,files in tqdm(islice(data_loader,0,early_stop)):
        for videos,labels,files in tqdm(data_loader):
            i = i+1
            if args.sequence_first:
                videos = videos.permute(2,0,1,3,4)
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            if args.arch == "I3D":
                loss = F.cross_entropy(torch.max(outputs, dim=2)[0],labels)
                outputs = torch.mean(outputs, 2)
            else:
                loss = criterion(outputs,labels)
            outputs_total.append(outputs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        accuracy = (correct / total) * 100.0
        print("///////////////")
        print('Accuracy = {}'.format( accuracy))
        print("Per class accuracy:")
        print(confusion_matrix.diag()/confusion_matrix.sum(1))
        print("///////////////",flush=True)
    return np.vstack(outputs_total)
