import numpy as np
import pandas as pd
import torch
from torch_sparse import coalesce
import pickle


def do(base_path,node_num,has_feature,is_dir):
    '''
    :param base_path:
    :param node_num:
    :param has_feature:
    :param is_dir:  是有向图
    :return:
    '''

    df=pd.read_csv(base_path+'/train.txt',sep=' ', index_col=None,header=None)
    node_set = set(df[1].append(df[2]))
    print('total_num =',node_num,'real_num=',len(node_set))
    old2new,new2old = {},{}
    for new,old in enumerate(node_set):
        old2new[old]=new
        new2old[new]=old

    edge_index,edge_type = get_train_sparse_adj(base_path+'/train.txt',old2new,is_dir)
    val_data = get_test_data(base_path+'/valid.txt',old2new)
    test_data = get_test_data(base_path+'/test.txt',old2new)
    if has_feature:
        feature_data = get_feature_data(base_path+'/feature.txt',new2old)
    else:
        feature_data = None

    all_data = {'old2new':old2new,
                'new2old':new2old,
                'edge_index':edge_index,
                'edge_type':edge_type,
                'feature_data':feature_data,
                'val_data':val_data,
                'test_data':test_data}
    torch.save(all_data,base_path+'/all_data.pkl')

def get_feature_data(path,new2old):
    df = pd.read_csv(path,sep = ' ',index_col=0,header=None,skiprows=[0])
    old_list = [new2old[i] for i in range(len(new2old))]
    df = df.loc[old_list]
    data = torch.from_numpy(df.to_numpy(dtype=np.float32))
    return data


def get_test_data(path,old2new):
    df = pd.read_csv(path, sep=' ', index_col=None, header=None)
    df[1].replace(old2new, inplace=True)
    df[2].replace(old2new, inplace=True)
    data = torch.from_numpy(df.to_numpy(dtype=np.int64))
    return data


def get_train_sparse_adj(path,old2new,is_dir):
    df = pd.read_csv(path, sep=' ', index_col=None, header=None)
    df[1].replace(old2new, inplace=True)
    df[2].replace(old2new, inplace=True)
    # 替换成功
    data = torch.from_numpy(df.to_numpy(dtype=np.int64))
    # [edge_type,row,col]
    type_num = data[:,0].max()
    N = len(old2new)
    self_loop_index = torch.stack([torch.arange(N),torch.arange(N)])
    self_loop_type = torch.zeros(N,dtype=torch.long)
    print('引入边，自环，num=',N)
    edge_index = [self_loop_index]
    edge_type = [self_loop_type]
    for type_id in range(1,type_num+1):
    #     对每类边施行反向，去重，操作
        index = data[:,0]==type_id
        i = data[index,1:].T
        v = torch.ones(i.shape[1],dtype=torch.long)
        if not is_dir:
            # 无向图化有向图
            i = torch.cat([i, i[[1, 0]]], dim=1)
            v = torch.cat([v, v], dim=0)
        # 去重
        i,v = coalesce(i,v,N,N)
        v[:] = type_id
        print('引入边，类别 %d，num= %d'%(type_id,v.shape[0]))
        edge_index.append(i)
        edge_type.append(v)
    edge_index = torch.cat(edge_index,dim=1)
    edge_type = torch.cat(edge_type,dim=0)
    print('训练集总边数：',edge_index.shape[1])
    # edge_index,edge_type = data[:, 1:].transpose(0, 1), data[:, 0]
    # train_adj = torch.sparse_coo_tensor(data[:, 1:].transpose(0, 1), data[:, 0])
    return edge_index,edge_type

if __name__ == '__main__':
    # base_path = '../data/amazon'
    # node_num = 10166
    # has_feature = True
    # is_dir = False

    # base_path = '../data/youtube'
    # node_num = 2000
    # has_feature = False
    # is_dir = False

    base_path = '../data/twitter'
    node_num = 10000
    has_feature = False
    is_dir = True
    do(base_path,node_num,has_feature,is_dir)