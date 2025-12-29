# Setup & Run Instructions

## 1. Configure WandB API Key

### Option A: Using `.env` file (Recommended)

1. **Get your WandB API key:**
   - Go to https://wandb.ai/authorize
   - Copy your API key

2. **Add to `.env` file:**
   ```bash
   # Edit .env in the repo root
   WANDB_API_KEY=your_actual_api_key_here
   ```

3. **Load the `.env` file before running training:**
   ```bash
   # In your shell session
   set -a
   source .env
   set +a
   ```

   Or add this to your SLURM script (e.g., `myscript.sh`):
   ```bash
   #!/bin/bash
   # ... SLURM directives ...
   
   # Load environment variables from .env
   set -a
   source .env
   set +a
   
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate sololearn
   cd learning/solo-learn/
   
   python main_pretrain.py \
       --config-path scripts/pretrain/cifar/ \
       --config-name simclr_original.yaml \
       ++name="simclr-og-cifar100" \
       ++data.dataset=cifar100 \
       ++checkpoint.dir="/path/to/checkpoints"
   ```

### Option B: Using Environment Variable (Alternative)

Set the API key directly in your shell:
```bash
export WANDB_API_KEY="your_actual_api_key_here"
```

### Option C: Disable WandB (For Quick Tests)

If you don't want to use WandB logging, add this to your training command:
```bash
python main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name simclr_original.yaml \
    ++wandb.enabled=False \
    ++data.dataset=cifar100
```

Or set offline mode:
```bash
export WANDB_MODE=offline
```

## 2. Run Training Examples

### Local Training (with WandB online logging)
```bash
# Load environment
set -a
source .env
set +a

conda activate sololearn

# Run from solo-learn directory
cd learning/solo-learn/
python main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name simclr_original.yaml \
    ++name="simclr-og-cifar100" \
    ++data.dataset=cifar100 \
    ++checkpoint.dir="/home/RUS_CIP/st190519/my_work/code/experiments/simclr_cifar_og"
```

### SLURM Job Submission
```bash
# Submit job with SLURM
sbatch myscript.sh
```

Make sure `myscript.sh` has the `.env` loader added as shown in Option A above.

## 3. Verify Setup

Check that WandB API key is loaded:
```bash
set -a
source .env
set +a
echo $WANDB_API_KEY
```

If it prints your key, you're ready to train!

## 4. Security Notes

⚠️ **Important:**
- `.env` is in `.gitignore` — it will NOT be committed to Git
- Never share your `.env` file or API key in code/logs
- If your key is exposed, regenerate it at https://wandb.ai/settings/keys
- Use different API keys for different machines/users if possible

## 5. Troubleshooting

**Error: "No API key configured"**
- Make sure you've set the API key in `.env`
- Ensure you ran `source .env` before training
- Check with `echo $WANDB_API_KEY`

**Error: "wandb.errors.errors.UsageError"**
- Run training with `++wandb.offline=True` to skip online logging
- Or disable WandB completely with `++wandb.enabled=False`

**Want to use offline mode permanently?**
- Edit `.env` and uncomment: `WANDB_MODE=offline`

**to train on HPC without the need to keep the terminal open**
```bash
nohup python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name simclr_original.yaml ++name="simclr-og2-cifar100" ++data.dataset=cifar100 ++checkpoint.dir="$HOME/my_work/code/experiments/simclr_cifar_og" > train.log 2>&1 &
```
```bash
## use these to monitor the training resource usage and output
watch -n 2 nvidia-smi
tail -f train.log
```
