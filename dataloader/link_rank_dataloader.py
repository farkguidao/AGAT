import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset, DataLoader
from utils.sparse_utils import *
from torch_sparse import coalesce
from dataloader.link_pre_dataloader import LinkPredictionDataloader
class LinkRankDataloader(LinkPredictionDataloader):
    pass