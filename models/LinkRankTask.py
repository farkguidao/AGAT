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
        self.val_result = {'mrr':-np.inf}

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        pos_edge, pos_edge_type, edge_id = batch
        em = self.get_em(mask=edge_id)  # type_num,N,d_model
        source = pos_edge[:, 0]
        target = pos_edge[:, 1]
        l1 = self.loss1(inputs=em[pos_edge_type, source], weights=self.w, labels=target,
                        neg_num=self.hparams.neg_num)
        self.log('loss1', l1, prog_bar=True)
        loss = l1
        if self.hparams.aggregator == 'agat':
            logits = (em[:, source] * self.w[target].unsqueeze(0)).sum(-1).T  # bs,t
            l2 = self.loss2(logits, pos_edge_type)
            self.log('loss2', l2, prog_bar=True)
            self.log('loss_all', l1 + l2, prog_bar=True)
            loss = loss + l2
        return loss

    def evalute(self,obj,pred,label):
        '''
        the code comes from compGCN
        :param pre: bs,
        :param label:
        :return:
        '''
        results={}
        b_range = torch.arange(pred.size()[0], device=self.device)
        target_pred = pred[b_range, obj]
        pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
        pred[b_range, obj] = target_pred
        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
        ranks = ranks.float()
        results['count'] = torch.numel(ranks)
        results['mr'] = torch.sum(ranks).item()
        results['mrr'] = torch.sum(1.0 / ranks).item()
        for k in range(10):
            results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)])
        return results

    def get_combined_results(self,left_results, right_results):
        '''
        the code comes from compGCN
        :param left_results:
        :param right_results:
        :return:
        '''
        results = {}
        count = float(left_results['count'])

        results['left_mr'] = round(left_results['mr'] / count, 5)
        results['left_mrr'] = round(left_results['mrr'] / count, 5)
        results['right_mr'] = round(right_results['mr'] / count, 5)
        results['right_mrr'] = round(right_results['mrr'] / count, 5)
        results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
        results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

        for k in range(10):
            results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
            results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
            results['hits@{}'.format(k + 1)] = round(
                (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
        return results

    def get_evalute_result(self,batch):
        '''
        :param batch: 前一半是 预测tail节点，得到left_result；后一半是预测head节点，得到right_result
        :return: results
        '''
        triple,label = batch
        bs = triple.shape[0]//2
        head,rel,tail = triple[:,0],triple[:,1],triple[:,2]
        em = self.get_em()
        pred = em[rel,head] @ self.w.T #bs*2,N

        left_reslut =   self.evalute(tail[:bs],pred[:bs],label[:bs])
        right_reslut =   self.evalute(tail[bs:],pred[bs:],label[bs:])
        result = self.get_combined_results(left_reslut,right_reslut)
        return result

    def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        result = self.get_evalute_result(batch)
        if self.val_result['mrr']<result['mrr']:
            self.val_result = result
        self.log_dict(result)


    def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        result = self.get_evalute_result(batch)
        self.test_result = result

    def on_test_end(self) -> None:
        with open(self.trainer.log_dir + '/best_result.txt', mode='w') as f:
            print('test_result:', self.test_result)
            f.write(str(self.test_result))

    def on_fit_end(self) -> None:
        with open(self.trainer.log_dir + '/best_val_result.txt', mode='w') as f:
            print('val_result:', self.val_result)
            f.write(str(self.val_result))