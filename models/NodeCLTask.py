from typing import Optional, Union, List

import torch
import torchmetrics
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
import numpy as np
from torch.optim import Optimizer

from models.AGAT import AGAT

class NodeClassificationTask(pl.LightningModule):
    def __init__(self, edge_index, edge_type, feature, N, aggregator, use_feature, feature_dim, d_model, type_num, L,
                 use_gradient_checkpointing, dropout, lr, wd):
        super(NodeClassificationTask, self).__init__()
        self.save_hyperparameters(ignore=['edge_index','edge_type','feature','N','degree'])
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type', edge_type)
        edge_type_num = edge_type.max()+1
        self.register_buffer('edge_feature', torch.eye(edge_type_num))
        self.fc_edge = nn.Linear(edge_type_num, d_model)

        if use_feature:
            self.register_buffer('feature',feature)
            self.fc_node = nn.Linear(feature_dim, d_model)
        else:
            self.feature = nn.Parameter(torch.randn(N,d_model))

        self.w = nn.Parameter(torch.FloatTensor(type_num, d_model))
        nn.init.xavier_uniform_(self.w)
        if aggregator == 'agat':
            self.agat = AGAT(type_num, d_model, L, use_gradient_checkpointing, dropout)
        elif aggregator == 'sgat':
            self.sgat = AGAT(1, d_model, L, use_gradient_checkpointing, dropout)

        self.loss =  nn.CrossEntropyLoss()
        self.max_macro_F1 = -np.inf
        self.max_micro_F1 = -np.inf
        self.micro_f1_cal = torchmetrics.F1(num_classes=type_num,average='micro',multiclass=True)
        self.macro_f1_cal = torchmetrics.F1(num_classes=type_num,average='macro',multiclass=True)

    def evalute(self,pre,label):
        micro_F1 = self.micro_f1_cal(pre,label)
        macro_F1 = self.macro_f1_cal(pre,label)
        if self.max_micro_F1 < micro_F1:
            self.max_micro_F1 = micro_F1
            self.max_macro_F1 = macro_F1
        self.log('micro-f1',micro_F1,prog_bar=True)
        self.log('macro-f1',macro_F1,prog_bar=True)
        self.micro_f1_cal.reset()
        self.macro_f1_cal.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.wd)
        return optimizer

    def get_em(self):
        if self.hparams.use_feature:
            feature = self.fc_node(self.feature)
        else:
            feature = self.feature
        edge_feature = self.fc_edge(self.edge_feature)
        if self.hparams.aggregator=='agat':
            em = self.agat(feature,self.edge_index,self.edge_type,edge_feature)
        elif self.hparams.aggregator=='sgat':
            em = self.sgat(feature,self.edge_index,self.edge_type,edge_feature)\
                .expand(self.hparams.type_num,feature.shape[0],self.hparams.d_model)
        return em #t,N,d_model

    def forward(self,node_id):
        em = self.get_em()
        node_em = em[:,node_id].transpose(0,1) #bs,t,d
        logits = (node_em * self.w).sum(-1) # bs,t
        return logits

    def training_step(self,batch, *args, **kwargs) -> STEP_OUTPUT:
        data = batch[0]
        node_id,label = data[:,0],data[:,1]
        pre = self(node_id)
        loss = self.loss(pre,label)
        self.log('loss',loss,prog_bar=True)
        return loss

    def validation_step(self,batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        data = batch[0]
        node_id, label = data[:, 0], data[:, 1]
        pre = self(node_id)
        self.evalute(pre,label)

    def test_step(self,batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.validation_step(batch)

    def on_fit_end(self) -> None:
        with open(self.trainer.log_dir + '/best_result.txt', mode='w') as f:
            result = {'micro-f1': float(self.max_micro_F1), 'macro-f1': float(self.max_macro_F1)}
            print('test_result:', result)
            f.write(str(result))

