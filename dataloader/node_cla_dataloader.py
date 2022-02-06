import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset, DataLoader
from utils.sparse_utils import *
from torch_sparse import coalesce
class NodeClassificationDataloader(pl.LightningDataModule):
    def __init__(self,datapath,batch_size,num_workers=0):
        super(NodeClassificationDataloader, self).__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.read_data()
    def read_data(self):
        data = torch.load(self.datapath)
        train_data, test_data, feature_data = data['train_data'], data['test_data'], data['feature_data']
        self.feature_data = feature_data
        self.test_dataset = TensorDataset(test_data)
        self.train_dataset = TensorDataset(train_data)
        self.edge_index, self.edge_type = data['edge_index'], data['edge_type']
        self.N, self.E = self.edge_index.max() + 1, self.edge_index.shape[1]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,self.batch_size,shuffle=True,num_workers=self.num_workers,drop_last=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,batch_size=len(self.test_dataset))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,batch_size=len(self.test_dataset))

if __name__ == '__main__':
    dataloader = NodeClassificationDataloader('../data/Aifb/all_data.pkl', 64)