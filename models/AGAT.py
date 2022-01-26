import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_softmax, scatter_add
from torch.utils.checkpoint import checkpoint
from torch_sparse import spmm

from dataloader.link_pre_dataloader import LinkPredictionDataloader


class AGAT(nn.Module):
    def __init__(self,type_num,d_model,L,use_gradient_checkpointing=False,dropout=.1):
        super(AGAT, self).__init__()
        self.type_num = type_num
        self.d_model = d_model
        self.L = L
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layer_list = nn.ModuleList([AGATLayer(type_num,d_model) for i in range(L)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for i in range(L)])
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x,edge_index,edge_type,edge_feature,mask=None):
        '''
        :param x: N,d_model
        :param path: E,d_model
        :param edge_index: 2,E
        :param mask:
        :return:
        '''
        N,d,E,eT = x.shape[0],x.shape[1],edge_type.shape[0],edge_feature.shape[0]
        x = x.expand(self.type_num,N,d)

        for i in range(self.L):
            x_ = x
            if self.use_gradient_checkpointing:
                x, edge_feature = checkpoint(self.layer_list[i],x,edge_index,edge_type,edge_feature,mask)
            else:
                x, edge_feature = self.layer_list[i](x,edge_index,edge_type,edge_feature,mask)
            if i == self.L-1:
                break
            x = x_+self.relu(self.dropout[i](x))
            edge_feature = self.relu(edge_feature)

        return x

class AGATLayer(nn.Module):
    def __init__(self,type_num,d_model):
        super(AGATLayer, self).__init__()
        self.type_num = type_num
        self.d_model = d_model
        self.theta_g = nn.Parameter(torch.FloatTensor(type_num, d_model))
        self.theta_hi = nn.Parameter(torch.FloatTensor(type_num, d_model))
        self.theta_hj = nn.Parameter(torch.FloatTensor(type_num, d_model))
        self.we = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.wr = nn.Parameter(torch.FloatTensor(d_model, d_model))
        nn.init.xavier_uniform_(self.theta_g)
        nn.init.xavier_uniform_(self.theta_hi)
        nn.init.xavier_uniform_(self.theta_hj)
        nn.init.xavier_uniform_(self.we)
        nn.init.xavier_uniform_(self.wr)
    def forward(self,x,edge_index,edge_type,edge_feature,mask):
        '''
        :param x:
        :param edge_index:
        :param edge_type:
        :param edge_feature:
        :param mask:
        :return:
        '''
        E = edge_type.shape[0]
        et=edge_feature.shape[0]
        T,N,d= x.shape
        theta_g, theta_hi, theta_hj, wr, we = self.theta_g,self.theta_hi,self.theta_hj,self.wr,self.we
        row, col = edge_index[0], edge_index[1]
        # 计算r_g分量
        r_g = (edge_feature.unsqueeze(0) * theta_g.unsqueeze(1)).sum(-1).index_select(1, edge_type)  # t,et->t,E
        r_hi = (x * theta_hi.unsqueeze(1)).sum(-1).index_select(1, row)  # t,N->t,E
        r_hj = (x * theta_hj.unsqueeze(1)).sum(-1).index_select(1, col)  # t,N->t,E
        r = r_g + r_hi + r_hj

        # h = x.index_select(1,col) # t,E,d
        # r = (torch.cat([path,h],dim=-1) * theta.unsqueeze(1)).sum(-1) #t,E
        if mask is not None:
            r = r.index_fill(1,mask,-np.inf)
            pass
        r = scatter_softmax(r, row, dim=-1)  # t,E
        edge_feature = edge_feature @ wr  # et,d_model
        if E>10*et*N:
            v_g = torch.sigmoid(edge_feature).view(1,et,1,d)
            v_h = (x @ we).view(T,1,N,d)
            v = (v_g*v_h)[:,edge_type,col] #T,E,d
        else:
            v_g = torch.sigmoid(edge_feature).index_select(0, edge_type).unsqueeze(0)  # 1,E,d_model
            v_h = (x @ we).index_select(1, col)
            v = v_g*v_h
        out = r.unsqueeze(-1) * v
        out = scatter_add(out, row, dim=-2)  # t,N,d_model
        return out, edge_feature

if __name__ == '__main__':
    dataloader = LinkPredictionDataloader('../data/amazon/all_data.pkl',64,64)
    edge_index,path = dataloader.edge_index,dataloader.edge_type
    E = path.shape[0]
    N = edge_index.max()+1
    # path = torch.randn(E,16)
    # feature = torch.randn(N,16)
    model = AGAT(4,64,3,True,0.1).cuda()
    # rs = model(feature.cuda(),path.cuda(),edge_index.cuda())

    x = torch.randn(N,64)
    edge_feature = torch.randn(3,64)
    rs = model(x.cuda(),edge_index.cuda(),path.cuda(),edge_feature.cuda())
