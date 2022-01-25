import os

import torch

from dataloader.link_pre_dataloader import LinkPredictionDataloader
from models.LinkPreTask import LinkPredictionTask
import pytorch_lightning as pl
import yaml
import argparse

def get_trainer_model_dataloader_from_yaml(yaml_path):
    with open(yaml_path) as f:
        settings = dict(yaml.load(f,yaml.FullLoader))

    dl = LinkPredictionDataloader(**settings['data'])
    model = LinkPredictionTask(dl.edge_index,dl.edge_type,dl.feature_data,dl.N, **settings['model'])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**settings['callback'])
    trainer = pl.Trainer(callbacks=[checkpoint_callback], **settings['train'])
    return trainer,model,dl


def train(parser):
    # dl=NSDataloader(batch_size=512*32)
    # model = M2GCNModel(N=dl.num_nodes,adj_list=dl.adj_list,lam=0.5)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='auc',mode='max')
    # trainer = pl.Trainer(max_epochs=10,callbacks=[checkpoint_callback],gpus=1,reload_dataloaders_every_n_epochs=1)
    # trainer.fit(model,dl)
    args = parser.parse_args()
    setting_path = args.setting_path
    trainer,model,dl = get_trainer_model_dataloader_from_yaml(setting_path)
    trainer.fit(model,dl)
    # 测试
    # 加载参数
    ckpt_path = trainer.log_dir + '/checkpoints/' + os.listdir(trainer.log_dir + '/checkpoints')[0]
    state_dict = torch.load(ckpt_path)['state_dict']
    model.load_state_dict(state_dict)
    trainer.test(model, dl.test_dataloader())
def test(parser):
    parser.add_argument('--ckpt_path',type=str,help='model checkpoint path')
    args = parser.parse_args()
    setting_path = args.setting_path
    trainer, model, dl = get_trainer_model_dataloader_from_yaml(setting_path)
    # 加载参数
    state_dict=torch.load(args.ckpt_path)['state_dict']
    model.load_state_dict(state_dict)
    trainer.test(model,dl.test_dataloader())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_path',type=str,default='settings/yot_settings.yaml')
    parser.add_argument("--test", action='store_true', help='test or train')
    temp_args, _ = parser.parse_known_args()
    if temp_args.test:
        test(parser)
    else:
        train(parser)