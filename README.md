# AGAT

Source code for paper "Aspect-Aware Graph Attention Network for Heterogeneous Information Networks"

## Requirements

The code has been tested under Python 3.8, with the following packages installed (along with their dependencies):

- torch >= 1.9.0
- pytorch-lightning >= 1.4.4
- torchmetrics >= 0.5.0
- torch-scatter >= 2.0.9
- torch-sparse >= 0.6.12
- numpy
- pandas
- tqdm
- yaml

## unzip

**Since git limits the size of a single file upload (<25M), we divide the datasets and the pre-trained models into multiple volumes. Please unzip the files in the directories `data`and`lightning_logs` first.**

```
cd ./data
sh do_unzip.sh
cd ../lightning_logs
sh do_unzip.sh
```

## Files in the folder

- **/data:** Store the dataset and prepared data.
- **/dataloader:** Codes of the dataloader.
- **/models:** Codes of the AGAT model , link-prediction task and semi-supervised classification task .
- **/utils:** Codes for data preparing and some other utils.
- **/lightning_logs:** Store the trained model parameters, setting files, checkpoints, logs and results.
- **main.py:** The main entrance of running.

## Basic usage

### Link Prediction Task

**train AGAT by**

```
# train AGAT .
python main.py --setting_path *.yaml

#  for example
#  youtube
python main.py --setting_path lightning_logs/youtube_best/yot_settings.yaml
# amazon
python main.py --setting_path lightning_logs/amazon_best/ama_settings.yaml
# twitter
python main.py --setting_path lightning_logs/twitter_best/tiw_settings.yaml

```
The `*.yaml` is the  configuration file.

And if you want to adjust the hyperparameters of the model, you can modify it in `*.yaml`, or create a similar configuration file, and specify `--setting_path` like this:

```
python main.py --setting_path yourpath.yaml
```

Checkpoints, logs, and results during training will be stored in the directory: `./lightning_logs/version_0`

And you can run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress.

**Load the pre-trained model and predict the test dataset by:**

```
# test 
python main.py --test --setting_path *.yaml --ckpt_path *.ckpt

# for example
#  youtube
python main.py --test --setting_path lightning_logs/youtube_best/yot_settings.yaml --ckpt_path lightning_logs/youtube_best/checkpoints/pre-trained.ckpt
# amazon
python main.py --test --setting_path lightning_logs/amazon_best/ama_settings.yaml --ckpt_path lightning_logs/amazon_best/checkpoints/pre-trained.ckpt
# twitter
python main.py --test --setting_path lightning_logs/twitter_best/tiw_settings.yaml --ckpt_path lightning_logs/twitter_best/checkpoints/pre-trained.ckpt
```
The result will be stored in the directory: `./lightning_logs/version_0`

If you want to load your trained model to predict the test data set, you only need to change `--setting_path` and `--ckpt_path`like this:

```
python main.py --test --setting_path yourpath.yaml --ckpt_path yourpath.ckpt
```

PS: Keep the configuration file unchanged during training and testing.

### Semi-supervised Classification Task

training and testing are similar to the Link Prediction Task.

**train:**

```
#  AIFB
python main.py --setting_path lightning_logs/aifb_best/aifb_settings.yaml

# PubMed
python main.py --setting_path lightning_logs/pub_best/pub_settings.yaml
```

**test:**
```
#  AIFB
python main.py --test --setting_path lightning_logs/aifb_best/aifb_settings.yaml --ckpt_path lightning_logs/aifb_best/checkpoints/pre-trained.ckpt

# PubMed
python main.py --test --setting_path lightning_logs/pub_best/pub_settings.yaml --ckpt_path lightning_logs/pub_best/checkpoints/pre-trained.ckpt
```
