from typing import Optional

import torch
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
import numpy as np

from models.AGAT import AGAT
from models.LinkPreTask import LinkPredictionTask
class LinkRankTask(LinkPredictionTask):

    def __init__(self, edge_index, edge_type, feature, N, aggregator, use_feature, feature_dim, d_model, type_num, L,
                 use_gradient_checkpointing, neg_num, dropout, lr, wd):
        super().__init__(edge_index, edge_type, feature, N, aggregator, use_feature, feature_dim, d_model, type_num, L,
                         use_gradient_checkpointing, neg_num, dropout, lr, wd)

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(batch, *args, **kwargs)

    def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        # 剔除不在训练集中的
        return super().validation_step(batch, *args, **kwargs)

    def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:

        return super().test_step(batch, *args, **kwargs)

    def on_test_end(self) -> None:
        super().on_test_end()

    def on_fit_end(self) -> None:
        super().on_fit_end()