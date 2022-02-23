# AGAT

Source code for paper "M2GCN: Multi-Modal Graph Convolutional Network for Polypharmacy Side Effects Discovery"

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
- **/models:** Codes of the AGAT model , link-prediction task and simi-node-classification task .
- **/utils:** Codes for data prepareing and some other utils.
- **/lightning_logs:** Store the trained model parameters, setting files, checkpoints, logs and results.
- **main.py:** The main entrance of running.

## Basic usage

### Train AGAT

train M2GCN by

```
python main.py
```

The default configuration file is `setting/settings.yaml`.

And if you want to adjust the hyperparameters of the model, you can modify it in `.setting/settings.yaml`, or create a similar configuration file, and specify `--setting_path` like this:

```
python main.py --setting_path yourpath.yaml
```

Checkpoints, logs, and results during training will be stored in the directory: `./lightning_logs/version_0`

And you can run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress.

### Link Prediction with pre-trained model

You can predict the interaction between drugs through the pre-trained model we provide.

**Since git limits the size of a single file upload (<25M), we divide the pre-trained model into multiple volumes. Please unzip the files in the directory `./lightning_logs/pre-trained/checkpoints/` first.**

Load the pre-trained model and predict the test dataset by:

```
python main.py --test --ckpt_path ./lightning_logs/pre-trained/checkpoints/pre-trained.ckpt
```

The result(auc,aupr) will be stored in the directory: `./lightning_logs/version_0`

If you want to load your trained model to predict the test data set, you only need to change `--ckpt_path`like this:

```
python main.py --test --ckpt_path yourpath.ckpt
```

PS: Keep the configuration file unchanged during training and testing.