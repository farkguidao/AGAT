import numpy as np
import pandas as pd
import torch
import pickle


def do(base_path,node_num,has_feature):

    df=pd.read_csv(base_path+'/train.txt',sep=' ', index_col=None,header=None)
    node_set = set(df[1].append(df[2]))
    print('total_num =',node_num,'real_num=',len(node_set))
    old2new,new2old = {},{}
    for new,old in enumerate(node_set):
        old2new[old]=new
        new2old[new]=old

    edge_index,edge_type = get_train_sparse_adj(base_path+'/train.txt',old2new)
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


def get_train_sparse_adj(path,old2new):
    df = pd.read_csv(path, sep=' ', index_col=None, header=None)
    df[1].replace(old2new, inplace=True)
    df[2].replace(old2new, inplace=True)
    # 替换成功
    # 存为稀疏型邻接矩阵
    data = torch.from_numpy(df.to_numpy(dtype=np.int64))
    edge_index,edge_type = data[:, 1:].transpose(0, 1), data[:, 0]
    # train_adj = torch.sparse_coo_tensor(data[:, 1:].transpose(0, 1), data[:, 0])
    return edge_index,edge_type

if __name__ == '__main__':
    # base_path = '../data/amazon'
    # node_num = 10166
    # has_feature = True

    base_path = '../data/youtube'
    node_num = 2000
    has_feature = False
    do(base_path,node_num,has_feature)