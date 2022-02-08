import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset, DataLoader
from utils.sparse_utils import *
from torch_sparse import coalesce
from dataloader.link_pre_dataloader import LinkPredictionDataloader
class LinkRankDataloader(LinkPredictionDataloader):

    def read_data(self):
        data = torch.load(self.datapath)
        # val_data,test_data,feature_data = data['val_data'],data['test_data'],data['feature_data']
        self.feature_data = None
        self.val_dataset = TensorDataset(data['val_triple'],data['val_label'])
        self.test_dataset = TensorDataset(data['test_triple'],data['test_label'])
        self.edge_index,self.edge_type = data['edge_index'],data['edge_type']
        self.N,self.E = data['p'].num_ent,self.edge_index.shape[1]
        self.edge_id = torch.arange(self.E)
        # mask除去自环
        mask = self.edge_type<(data['p'].num_rel*2)
        self.train_dataset = TensorDataset(self.edge_index.T[mask],self.edge_type[mask],self.edge_id[mask])