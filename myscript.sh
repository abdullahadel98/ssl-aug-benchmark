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
    ++name="simclr-og-cifar100" \
    ++data.dataset=cifar100 \
    ++checkpoint.dir="/cig/common06nb/ssp_interns/Abdullah_Abdelaal/models/ssl-aug/simclr_cifar_og"


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

