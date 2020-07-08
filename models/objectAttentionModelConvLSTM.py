
import torch
from models import resnetMod 
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from models.MyConvLSTMCell import *
import torchvision

class attentionModel(nn.Module):
    def __init__(self,dropout=0.0, num_classes=61, mem_size=512, arch='resnet34',GPU=0):
        super(attentionModel, self).__init__()
#        device = torch.device("cuda:"+ str(GPU)) if torch.cuda.is_available() else 'cpu'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        if arch=='resnet34':
            self.resNet = resnetMod.resnet34(pretrained=True, noBN=True, GPU = GPU)
            self.lstm_cell = MyConvLSTMCell(512, mem_size) #works for resnet34
        elif arch=='densenet161':
            self.resNet = resnetMod.densenet161(pretrained=True, noBN=False, GPU = GPU)
            self.lstm_cell = MyConvLSTMCell(2208, mem_size) #works for densenet161
        elif arch=='senet154':
            self.resNet = resnetMod.senet154(pretrained=True, noBN=False, GPU = GPU)
            self.lstm_cell = MyConvLSTMCell(2048, mem_size) #works for senet154

        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable,mean_model=None):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(self.device)),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(self.device)))
        if mean_model is not None:
            attention_map_list = mean_model.get_attention_map(inputVariable)

        for t in range(inputVariable.size(0)):
            #print inputVariable[t].size()

            inputVariable2 = inputVariable[t]
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable2)
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            #print("weight_softmax dim ",feature_conv1.size())
            #print("weight_softmax dim ",self.weight_softmax.size())

            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            if mean_model is not None:
                attentionMAP = attention_map_list[t]
            else:
                attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            state = self.lstm_cell(attentionFeat, state)
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        
#        return feats, feats1
        return feats


    def get_attention_map(self, inputVariable):
        attention_map_list = []
        for t in range(inputVariable.size(0)):
            if self.DecoFlag:
                inputVariable2 = self.DECO(inputVariable[t])
            else:
                inputVariable2 = inputVariable[t]
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable2)
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            #print("weight_softmax dim ",feature_conv1.size())
            #print("weight_softmax dim ",self.weight_softmax.size())

            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attention_map_list.append(attentionMAP)        
        return attention_map_list
