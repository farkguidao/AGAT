from typing import Optional

import torch
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
import numpy as np

from models.AGAT import AGAT


class LinkPredictionTask(pl.LightningModule):
    def __init__(self,edge_index,edge_type,feature,N,degree,use_feature,feature_dim,d_model,type_num, L,use_gradient_checkpointing,neg_num,dropout,lr,wd):
        super(LinkPredictionTask, self).__init__()
        # 工程类组件
        self.save_hyperparameters(ignore=['edge_index','edge_type','feature','N','degree'])
        self.register_buffer('edge_index',edge_index)
        self.register_buffer('edge_type',edge_type)
        self.register_buffer('edge_feature',torch.eye(type_num+1))
        if use_feature:
            self.register_buffer('feature',feature)
            self.fc_node = nn.Linear(feature_dim, d_model)
        else:
            self.feature = nn.Parameter(torch.randn(N,d_model))
            # nn.init.xavier_uniform_(self.feature)
        self.loss2 = nn.CrossEntropyLoss()
        self.loss1 = NCELoss(N,degree)
        self.val_best_auc = 0
        self.val_best_aupr = 0
        self.test_best_auc = 0
        self.test_best_aupr = 0
        #

        self.fc_edge = nn.Linear(type_num+1,d_model)
        self.w = nn.Parameter(torch.FloatTensor(N,d_model))
        nn.init.xavier_uniform_(self.w)
        self.agat = AGAT(type_num,d_model,L,use_gradient_checkpointing,dropout)

    def get_em(self,mask=None):
        if self.hparams.use_feature:
            feature = self.fc_node(self.feature)
        else:
            feature = self.feature
        edge_feature = self.fc_edge(self.edge_feature)
        em = self.agat(feature,self.edge_index,self.edge_type,edge_feature,mask)
        return em

    def training_step(self, batch,*args, **kwargs) -> STEP_OUTPUT:
        pos_edge,pos_edge_type,edge_id = batch
        em = self.get_em(mask=edge_id) #type_num,N,d_model
        source = pos_edge[:,0]
        target = pos_edge[:,1]
        l1 = self.loss1(inputs=em[pos_edge_type-1,source],weights=self.w,labels=target,neg_num=self.hparams.neg_num)
        # em[:,source] #t,bs,d
        # self.w[target].unsqueeze(0) #1, bs,d
        # (em[:, source] * self.w[target].unsqueeze(0)).sum(-1) #t,bs
        logits = (em[:, source] * self.w[target].unsqueeze(0)).sum(-1).T # bs,t
        l2 = self.loss2(logits,pos_edge_type-1)
        self.log('loss1', l1, prog_bar=True)
        self.log('loss2', l2, prog_bar=True)
        self.log('loss_all', l1+l2, prog_bar=True)
        return l1+l2

    def validation_step(self, batch,*args, **kwargs) -> Optional[STEP_OUTPUT]:
        em = self.get_em()
        data = batch[0]
        edge_type,source,target,label = data[:,0],data[:,1],data[:,2],data[:,3]
        score = (em[edge_type-1,source] * self.w[target]).sum(-1) #bs
        score = torch.sigmoid(score)
        auc = torchmetrics.functional.auroc(score, label, pos_label=1)
        aupr = torchmetrics.functional.average_precision(score, label, pos_label=1)
        if auc > self.val_best_auc:
            self.val_best_auc = auc
            self.val_best_aupr = aupr
        self.log('val_auc', auc, prog_bar=True)
        self.log('val_aupr', aupr, prog_bar=True)

    def test_step(self, batch,*args, **kwargs) -> Optional[STEP_OUTPUT]:
        em = self.get_em()
        data = batch[0]
        edge_type, source, target, label = data[:,0], data[:,1], data[:,2], data[:,3]
        score = (em[edge_type - 1, source] * self.w[target]).sum(-1)  # bs
        score = torch.sigmoid(score)
        auc = torchmetrics.functional.auroc(score, label, pos_label=1)
        aupr = torchmetrics.functional.average_precision(score, label, pos_label=1)
        if auc > self.test_best_auc:
            self.test_best_auc = auc
            self.test_best_aupr = aupr
    def on_test_end(self) -> None:
        with open(self.trainer.log_dir + '/best_result.txt', mode='w') as f:
            result = {'auc': float(self.test_best_auc), 'aupr': float(self.test_best_aupr)}
            print('test_result:', result)
            f.write(str(result))
    # 结束时存储最优结果
    def on_fit_end(self) -> None:
        with open(self.trainer.log_dir + '/val_best_result.txt', mode='w') as f:
            result = {'auc': float(self.val_best_auc), 'aupr': float(self.val_best_aupr)}
            print('val_best_result:', result)
            f.write(str(result))



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.wd)
        return optimizer

class NCELoss(nn.Module):
    def __init__(self,N,degree):
        super(NCELoss, self).__init__()
        self.N = N
        self.register_buffer('degree',degree)
        self.bce=nn.BCEWithLogitsLoss()
    def forward(self,inputs,weights,labels,neg_num):
        # neg_batch = torch.randint(0, self.N, (neg_num*inputs.shape[0],),
        #                           dtype=torch.long,device=inputs.device)
        neg_batch = torch.multinomial(self.degree,neg_num*inputs.shape[0],True)
        target = weights[torch.cat([labels,neg_batch],dim=0)]
        label = torch.zeros(target.shape[0],device=inputs.device)
        label[:labels.shape[0]]=1
        # bs,d_model-> bs*(neg_num+1),d_model
        source = inputs.repeat((neg_num+1,1))
        return self.bce((source*target).sum(dim=-1),label)

