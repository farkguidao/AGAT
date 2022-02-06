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
    edge_index,edge_type = get_train_sparse_adj(base_path+'/link.dat',is_dir)
    train_data = get_label_data(base_path+'/label.dat')
    test_data = get_label_data(base_path+'/label.dat.test')
    if has_feature:
        feature_data = get_feature_data(base_path+'/node.dat')
    else:
        feature_data = None

    all_data = {'edge_index':edge_index,
                'edge_type':edge_type,
                'feature_data':feature_data,
                'train_data':train_data,
                'test_data':test_data}
    torch.save(all_data,base_path+'/all_data.pkl')

def get_feature_data(path):
    node_df = pd.read_csv(path, sep='\t', header=None, quoting=3)
    dd = node_df[3].str.split(',', expand=True).astype(np.float32)
    data = torch.from_numpy(dd.to_numpy(dtype=np.float32))
    return data


def get_label_data(path):
    df = pd.read_csv(path, sep='\t', index_col=None, header=None)
    df = df[[0,3]]
    data = torch.from_numpy(df.to_numpy(dtype=np.int64))
    return data


def get_train_sparse_adj(path,is_dir):
    df = pd.read_csv(path, sep='\t', index_col=None, header=None)
    # 替换成功
    data = torch.from_numpy(df.to_numpy(dtype=np.int64))
    data = data[:,[2,0,1]] # [edge_type,row,col]
    type_num = data[:,0].max()
    N = data[:,1:].max()+1
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

    # base_path = '../data/PubMed'
    # node_num = 63109
    # has_feature = True
    # is_dir = True

    base_path = '../data/Aifb'
    node_num = 8285
    has_feature = False
    is_dir = True
    do(base_path,node_num,has_feature,is_dir)