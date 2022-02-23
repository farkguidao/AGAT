import os

import torch
import pytorch_lightning as pl
import yaml
from dataloader.link_pre_dataloader import LinkPredictionDataloader
from dataloader.node_cla_dataloader import NodeClassificationDataloader
from models.LinkPreTask import LinkPredictionTask
from models.NodeCLTask import NodeClassificationTask

TASK = {
    'link_pre':(LinkPredictionDataloader,LinkPredictionTask),
    'simi_node_CL':(NodeClassificationDataloader,NodeClassificationTask)
}
# 用来在晚上连续跑实验的工具
def get_trainer_model_dataloader_from_dir(settings):
    DATALOADER, MODEL = TASK[settings['task']]
    dl = DATALOADER(**settings['data'])
    model = MODEL(dl.edge_index, dl.edge_type, dl.feature_data, dl.N, **settings['model'])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**settings['callback'])
    trainer = pl.Trainer(callbacks=[checkpoint_callback], **settings['train'])
    return trainer, model, dl

def plan(base_settings,model_replace_key,model_replace_values):
    '''
    :param base_settings: 基础配置
    :param model_replace_key: 取代的超参
    :param model_replace_values: 超参值的列表
    :return:
    '''
    for v in model_replace_values:
        base_settings['model'][model_replace_key] = v
        print('--------------------------------------------------')
        print(model_replace_key, '=', v, 'has bean done!')
        trainer,model,dl=get_trainer_model_dataloader_from_dir(base_settings)
        trainer.fit(model,dl)
        # 测试
        # 加载参数
        ckpt_path = trainer.log_dir + '/checkpoints/' + os.listdir(trainer.log_dir + '/checkpoints')[0]
        state_dict = torch.load(ckpt_path)['state_dict']
        model.load_state_dict(state_dict)
        trainer.test(model, dl.test_dataloader())
        print(model_replace_key, '=', v, 'has finished! result in',trainer.log_dir)
        print('--------------------------------------------------')
        del trainer
        del model
        del dl
    print('finish plan!')

if __name__ == '__main__':
    yaml_path = '../settings/wn_settings.yaml'
    # key = 'L'
    # values = [1,2,3,4,5,6]
    key = 'lambed'
    values = [10,1,0.1,0.01,0.001,0]
    with open(yaml_path) as f:
        settings = dict(yaml.load(f,yaml.FullLoader))
    plan(settings,key,values)
