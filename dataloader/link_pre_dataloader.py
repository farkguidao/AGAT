import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset, DataLoader
from utils.sparse_utils import *
from torch_sparse import coalesce
class LinkPredictionDataloader(pl.LightningDataModule):
    def __init__(self,datapath,batch_size,num_workers=0):
        super(LinkPredictionDataloader, self).__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.read_data()
    def read_data(self):
        data = torch.load(self.datapath)
        val_data,test_data,feature_data = data['val_data'],data['test_data'],data['feature_data']
        self.feature_data = feature_data
        self.val_dataset = TensorDataset(val_data)
        self.test_dataset = TensorDataset(test_data)
        # # 自动去重
        # i,v = data['edge_index'],data['edge_type']
        # if not self.is_dir:
        #     # 无向图化有向图
        #     i = torch.cat([i,i[[1,0]]],dim=1)
        #     v = torch.cat([v,v],dim=0)
        # n = 1+i.max()
        # i,v = coalesce(i,v,n,n,op='max')
        # train_adj = torch.sparse_coo_tensor(i,v).coalesce()
        # # 添加自环0
        # train_adj = (train_adj + sparse_diags([0]*train_adj.shape[0])).coalesce()

        # self.train_adj = train_adj
        # self.edge_index,self.edge_type,self.edge_id = train_adj.indices(),train_adj.values(),torch.arange(train_adj._nnz())
        # self.feature_data = feature_data

        self.edge_index,self.edge_type = data['edge_index'],data['edge_type']
        self.N,self.E = self.edge_index.max()+1,self.edge_index.shape[1]
        self.edge_id = torch.arange(self.E)
        # mask除去自环
        mask = self.edge_type>0
        self.train_dataset = TensorDataset(self.edge_index.T[mask],self.edge_type[mask],self.edge_id[mask])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,self.batch_size,shuffle=True,num_workers=self.num_workers,drop_last=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,batch_size=len(self.test_dataset))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset,batch_size=len(self.val_dataset))


if __name__ == '__main__':
    dataloader = LinkPredictionDataloader('../data/amazon/all_data.pkl',64)
