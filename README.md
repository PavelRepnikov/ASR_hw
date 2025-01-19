# Automatic Speech Recognition (ASR) with PyTorch

W&B report link: https://wandb.ai/rtfiof-hse-university/ASR_hw/reports/asr-hw-report--VmlldzoxMDk5NjczNA?accessToken=hchakcysarclwf490nirjnvapg1sdgfuwmd3cs0coqfnzy16y4x9uwhekanp4elc

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

Automatic Speech Recognition model DeepSpeech2 implemented from scratch in PyTorch. 

This repository contains a project on Automatic Speech Recognition (ASR) with all necessary scripts and instructions for training and infering the model.

## Installation

Follow these steps to install the project:

1. (Optional) Create and activate new environment using `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).
   `venv` (`+pyenv`) version:

   ```bash
   # create env
   conda create -n check python=3.10

   # activate env
   conda activate check
   ```

2. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Train

To train a model, log in to wandb and run the following commands:

1. First,
   
```bash
python train.py -cn=config-1 
```
for 94 epochs

2. Second, train it with
```bash
python train.py -cn=config-3
```

3. Third,
```bash
python train.py -cn=config-4
```

Note that paths to the models may differ in your system

# How To Evaluate

1. To download the needed files to eval my model:

```bash
python download_checkpoint.py
```

2. To run inference:

```bash
python inference.py -cn=inference
```


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
