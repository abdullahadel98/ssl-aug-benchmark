
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

nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name simclr_trivAug.yaml  ++name="simclr-trivaug-coloraug-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/simclr_cifar_trivaug_coloraug" ++devices=[0] > train_simclr_trivaug2.log 2>&1 &

## experiment 4 byol with trivial augment
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name byol_trivAug.yaml  ++name="byol-trivaug-coloraug-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/byol_cifar_trivaug_coloraug" ++devices=[1] > train_byol_trivaug.log 2>&1 &


nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name dino_trivAug.yaml  ++name="dino-trivaug-coloraug-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/dino_cifar_trivaug_coloraug" ++devices=[0] > train_dino_trivaug.log 2>&1 &


nohup python main_pretrain.py --config-path scripts/pretrain/mvtec-ad/ --config-name simclr.yaml ++name="simclr-mvtec-ad" ++data.dataset=mvtec-ad ++data.train_path="/home/RUS_CIP/st190519/my_work/code/ssl-aug-benchmark/learning/draem/datasets/mvtec" ++data.val_path="/home/RUS_CIP/st190519/my_work/code/ssl-aug-benchmark/learning/draem/datasets/mvtec" ++checkpoint.dir="$HOME/my_work/code/experiments/simclr_mvtec_ad" > train_simclr_mvtec.log 2>&1 &

## cifar style transfer simclr
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name simclr_styletrans.yaml  ++name="simclr-styletrans25per-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/simclr_cifar_styletrans25per" ++devices=[1] > train_simclr_styletrans.log 2>&1 &

## cifar style transfer byol
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name byol_styletrans.yaml  ++name="byol-styletrans30per-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/byol_cifar_styletrans30per" ++devices=[0] > train_byol_styletrans.log 2>&1 &

## cifar style transfer dino
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name dino_styletrans.yaml  ++name="dino-styletrans10per-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/dino_cifar_styletrans10per" ++devices=[0] > train_dino_styletrans.log 2>&1 &

## cifar solo simclr
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name simclr_styletrans.yaml  ++name="simclr-solo-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/simclr_cifar_solo" ++devices=[1] > train_simclr_solo.log 2>&1 &

### cifar solo byol
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name byol_styletrans.yaml  ++name="byol-solo-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/byol_cifar_solo" ++devices=[1] > train_byol_solo.log 2>&1 &

### cifar solo dino
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name dino_styletrans.yaml  ++name="dino-solo-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/dino_cifar_solo" ++devices=[0] > train_dino_solo.log 2>&1 &

############################################################################################
# linear evaluation
############################################################################################

nohup python main_linear.py --config-path scripts/linear/cifar-100 --config-name simclr ++pretrained_feature_extractor="$HOME/my_work/code/experiments/byol_cifar_styletrans10per/byol/kjwban1k/" ++name="simclr-styletrans10per-cifar100" ++checkpoint.dir="$HOME/my_work/code/experiments/linear/simclr_cifar_styletrans10per" ++devices=[1] > test_simclr_styletrans.log 2>&1 &


############################################################################################
# knn evaluation
############################################################################################

python main_knn.py \
    --dataset cifar100 \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --batch_size 128 \
    --num_workers 4 \
    --pretrained_checkpoint_dir $HOME/my_work/code/experiments/byol_cifar_styletrans10per/byol/kjwban1k/ \
    --k 1 5 10 20 50 100 200 \
    --temperature 0.01 0.05 0.1 0.2 0.5 \
    --feature_type backbone projector \
    --distance_function euclidean cosine 

set -a
source .env
set +a

conda activate sololearn

# Run from solo-learn directory
cd learning/solo-learn/

watch -n 2 nvidia-smi

nohup python train_DRAEM.py --gpu_id 1 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path $HOME/my_work/code/experiments/draem_mvtec_og --log_path ./logs/ > train_draem.log 2>&1 &

nohup python train_DRAEM.py --gpu_id 1 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path $HOME/my_work/code/experiments/draem_mvtec_og --log_path ./logs/ --visualize True > train_draem.log 2>&1 &


# Basic: 4 CIFAR-10 images with comparison visualization
python visualize_style_transfer.py

# Specific dataset with more images
python visualize_style_transfer.py --dataset cifar100 --num-images 6

# Save individual images to disk
python visualize_style_transfer.py --num-images 4 --save-individual

# Adjust style strength and probability
python visualize_style_transfer.py --alpha 0.8 --probability 0.5

# Use real VGG models if available
python visualize_style_transfer.py --use-real-models --num-images 4

# Combine options
python visualize_style_transfer.py --dataset cifar100 --num-images 6 --save-individual --alpha 0.9