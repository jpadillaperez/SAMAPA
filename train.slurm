#!/bin/sh
#SBATCH --job-name=SAMAPA  # Job name
#SBATCH --output=output/logs/SAMAPA-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=output/logs/SAMAPA-%A.err  # Standard error of the script
#SBATCH --time=3-24:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1 # --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=1  # Number of CPUs (Don't use more than 12 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48G per GPU unless you absolutely need it and know what you are doing)

CURRENT_LOG_DIR=output/logs
OLD_LOG_DIR=output/logs/previous_logs

# Move old logs to the new directory except for the current log file
for file in $CURRENT_LOG_DIR/*.out $CURRENT_LOG_DIR/*.err; do
    if [[ $file != $CURRENT_LOG_DIR/SAMAPA-$SLURM_JOB_ID.out ]] && [[ $file != $CURRENT_LOG_DIR/SAMAPA-$SLURM_JOB_ID.err ]]; then
        mv "$file" "$OLD_LOG_DIR/"
    fi
done

# load python module
source /home/guests/jorge_padilla/miniconda3/etc/profile.d/conda.sh

# Set your wandb API key
export WANDB_API_KEY=$(cat ./wandb_api_key.txt)

# activate corresponding environment
conda deactivate 
conda activate samapa

python -u ./train.py

