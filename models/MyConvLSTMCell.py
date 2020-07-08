
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch as F

class MyConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size=3, stride=1, padding=1,GPU=0):
        super(MyConvLSTMCell, self).__init__()
        #sdevice = torch.device("cuda:"+ str(GPU)) 
        self.GPU=GPU
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_i_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.conv_i_xx  = self.conv_i_xx.cuda(GPU)
        self.conv_i_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)
        
        #self.conv_i_hh =  self.conv_i_hh.cuda(GPU)
        self.conv_f_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_c_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_c_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_o_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        torch.nn.init.xavier_normal_(self.conv_i_xx.weight)
        torch.nn.init.constant_(self.conv_i_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_i_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_f_xx.weight)
        torch.nn.init.constant_(self.conv_f_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_f_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_c_xx.weight)
        torch.nn.init.constant_(self.conv_c_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_c_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_o_xx.weight)
        torch.nn.init.constant_(self.conv_o_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_o_hh.weight)

    def forward(self, x, state):
        #device = torch.device("cuda:"+ str(self.GPU)) 
        if state is None:
            state = (Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).to(device) ),
                     Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).to(device) ) )
        ht_1, ct_1 = state
        #ht_1 = ht_1.to(device) 
        #ct_1 = ct_1.to(device)
        #x = x.to(device)
        it = F.sigmoid(self.conv_i_xx(x) + self.conv_i_hh(ht_1))
        ft = F.sigmoid(self.conv_f_xx(x) + self.conv_f_hh(ht_1))
        ct_tilde = F.tanh(self.conv_c_xx(x) + self.conv_c_hh(ht_1))
        #print(ct_tilde.size())
        #print(ct_1.size())
        ct = (ct_tilde * it) + (ct_1 * ft)
        ot = F.sigmoid(self.conv_o_xx(x) + self.conv_o_hh(ht_1))
        ht = ot * F.tanh(ct)
        return ht, ct