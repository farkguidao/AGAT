import torch
from torch import nn
from torch_scatter import scatter_softmax, scatter_add
from torch_sparse import spmm

from dataloader.link_pre_dataloader import LinkPredictionDataloader


class AGAT(nn.Module):
    def __init__(self,type_num,d_model,L,dropout=.1):
        super(AGAT, self).__init__()
        self.type_num = type_num
        self.d_model = d_model
        self.L = L
        self.theta = nn.ParameterList()
        self.we = nn.ParameterList()
        self.wr = nn.ParameterList()
        self.dropout = nn.ModuleList()
        for i in range(L):
            theta = nn.Parameter(torch.FloatTensor(type_num,3*d_model))
            we = nn.Parameter(torch.FloatTensor(d_model,d_model))
            wr = nn.Parameter(torch.FloatTensor(d_model,d_model))
            nn.init.xavier_uniform_(theta)
            nn.init.xavier_uniform_(we)
            nn.init.xavier_uniform_(wr)
            self.theta.append(theta)
            self.we.append(we)
            self.wr.append(wr)
            self.dropout.append(nn.Dropout(dropout))
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
        # edge_feature = edge_feature.expand(self.type_num,eT,d)
        for i in range(self.L):
            x,edge_feature = self.layer_forward(x,edge_index,edge_type,edge_feature,mask,
                                                self.theta[i][:,:d],self.theta[i][:,d:2*d],self.theta[i][:,2*d:],self.wr[i],self.we[i])
            if i == self.L-1:
                break
            x = self.dropout[i](self.relu(x))
            edge_feature = self.relu(edge_feature)
        return x

    def layer_forward(self,x,edge_index,edge_type,edge_feature,mask,theta_g,theta_hi,theta_hj,wr,we):
        '''
        :param x: type_num,N,d_model
        :param edge_index:
        :param edge_type:
        :param edge_feature: edge_type,d_model
        :param mask:
        :param theta:
        :param wr:
        :param we:
        :return:
        '''
        row,col = edge_index[0],edge_index[1]
        # 计算r_g分量
        r_g = (edge_feature.unsqueeze(0) * theta_g.unsqueeze(1)).sum(-1).index_select(1,edge_type) #t,et->t,E
        r_hi = (x * theta_hi.unsqueeze(1)).sum(-1).index_select(1,row) # t,N->t,E
        r_hj = (x * theta_hj.unsqueeze(1)).sum(-1).index_select(1,col) # t,N->t,E
        r = r_g+r_hi+r_hj

        # h = x.index_select(1,col) # t,E,d
        # r = (torch.cat([path,h],dim=-1) * theta.unsqueeze(1)).sum(-1) #t,E
        if mask is not None:
            pass
        r = scatter_softmax(r,row,dim=-1) #t,E
        edge_feature = edge_feature @ wr # et,d_model
        v_g = torch.sigmoid(edge_feature).index_select(0,edge_type).unsqueeze(0) #1,E,d_model
        v_h = (x @ we).index_select(1,col)
        out = r.unsqueeze(-1)*v_g*v_h
        out = scatter_add(out, row, dim=-2)# t,N,d_model
        return out, edge_feature

if __name__ == '__main__':
    dataloader = LinkPredictionDataloader('../data/amazon/all_data.pkl',64,64)
    edge_index,path = dataloader.edge_index,dataloader.edge_type
    E = path.shape[0]
    N = edge_index.max()+1
    # path = torch.randn(E,16)
    # feature = torch.randn(N,16)
    model = AGAT(4,32,3,0.1).cuda()
    # rs = model(feature.cuda(),path.cuda(),edge_index.cuda())
    x = torch.randn(N,32)
    edge_feature = torch.randn(3,32)
    rs = model(x.cuda(),edge_index.cuda(),path.cuda(),edge_feature.cuda())
