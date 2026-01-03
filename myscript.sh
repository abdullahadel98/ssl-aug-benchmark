#!/bin/bash
#SBATCH --job-name=simclr_og_train
#SBATCH --error=simclr_og_train.err.log
#SBATCH --partition=sharedhiti,shared,student
#SBATCH --gres=gpu:1,gpu_mem:48000
#SBATCH --output=simclr_og_train.out.log
#SBATCH --mem=32G
#SBATCH --time=03-00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=abdullahadelmohammed.abdelaal@sony.com

echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sololearn
cd learning/solo-learn/
# path to training script folder
# training config name
# add new arguments (e.g. those not defined in the yaml files)
# by doing ++new_argument=VALUE
# pytorch lightning's arguments can be added here as well.
python main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name simclr_original.yaml \
    ++name="simclr-og2-cifar100" \
    ++data.dataset=cifar100 \
    ++checkpoint.dir="~/my_work/code/experiments/simclr_cifar_og"


# run = wandb.init(
#     # Set the wandb entity where your project will be logged (generally your team name).
#     entity="my-awesome-team-name",
#     # Set the wandb project where this run will be logged.
#     project="my-awesome-project",
#     # Track hyperparameters and run metadata.
#     config={
#         "learning_rate": 0.02,
#         "architecture": "CNN",
#         "dataset": "CIFAR-100",
#         "epochs": 10,
#     },
# )

## experiment 2 dino
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name dino_original.yaml ++name="dino-og-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/dino_cifar_og" > train_dino_og.log 2>&1 &

## experiment 3 simclr with trivial augment
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name simclr.yaml ++name="simclr2-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/simclr_cifar_2" > train_simclr_trivaug.log 2>&1 &

nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name simclr_trivAug.yaml  ++name="simclr-randaug-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/simclr_cifar_randaug" ++devices=[0] > train_simclr_trivaug2.log 2>
&1 &

## experiment 4 byol with trivial augment
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name byol_trivAug.yaml  ++name="byol-randaug-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/byol_cifar_randaug" ++devices=[0] > train_byol_trivaug.log 2>
&1 &

nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name dino_trivAug.yaml  ++name="dino-randaug-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/dino_cifar_randaug" > train_dino_trivaug.log 2>
&1 &


set -a
source .env
set +a

conda activate sololearn

# Run from solo-learn directory
cd learning/solo-learn/


nohup python train_DRAEM.py --gpu_id 1 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path $HOME/my_work/code/experiments/draem_mvtec_og --log_path ./logs/ > train_draem.log 2>&1 &

nohup python train_DRAEM.py --gpu_id 1 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path $HOME/my_work/code/experiments/draem_mvtec_og --log_path ./logs/ --visualize True > train_draem.log 2>&1 &