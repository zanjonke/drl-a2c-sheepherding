# Deep reinforcement learning method for the shepherding problem

## Project description

This project is an extension of the deep reinforcement learning for sheepherding project. In this project we fully intergrate the sheepherding environment with the openAI 'gym' package and use the 'ptan' as an interface between the model and the evironment. In this project we tried to produce an end-to-end network which was able to solve the environment regardles of starting conditions.

## Repository description

The main files and folders are the following:

```bash
.
├── train.py
├── sheepherding.py
├── train_single_gpu.sh
├── lib
└── ptan
```

Where

- **train.py** implements the training framework (actor-critic network with a ResNet18 backbone along with the training pipeline)
- **sheepherding.py** implements the Strombom et. al. sheepherding environment
- **train_single_gpu.sh** implements the configuration to be able to run the project within a super computer that uses SLURM
- **ptan** which is a modified version of the ptan library (to fir our purposes)
- **lib** implements some auxiliary function

## Repository description

First install the required packages by using

`pip install -r requirements.txt`

After the installation, the files can be run.

`python sheepherding.py`

to run the sheepherding environment with random actions.

`python train.py --name <name> --cuda`

to run the training pipeline. The <name> parameter is the folder where the progress of the training will be stored and the optional 'cuda' flag can be passed in order to use a gpu.
