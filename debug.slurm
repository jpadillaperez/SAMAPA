#!/bin/sh
#SBATCH --job-name=SAMAPA-DEBUG  # Job name
#SBATCH --output=output/logs/SAMAPA-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=output/logs/SAMAPA-%A.err  # Standard error of the script
#SBATCH --time=2-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=1  # Number of CPUs (Don't use more than 12 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48G per GPU unless you absolutely need it and know what you are doing)

CURRENT_LOG_DIR=/home/guests/jorge_padilla/code/Guided_Research/SAMAPA/output/logs
OLD_LOG_DIR=/home/guests/jorge_padilla/code/Guided_Research/SAMAPA/output/logs/previous_logs

# Move old logs to the new directory except for the current log file
for file in $CURRENT_LOG_DIR/*.out $CURRENT_LOG_DIR/*.err; do
    if [[ $file != $CURRENT_LOG_DIR/SAMAPA-$SLURM_JOB_ID.out ]] && [[ $file != $CURRENT_LOG_DIR/SAMAPA-$SLURM_JOB_ID.err ]]; then
        mv "$file" "$OLD_LOG_DIR/"
    fi
done

# load python module
source /home/guests/jorge_padilla/miniconda3/etc/profile.d/conda.sh

# activate corresponding environment
conda deactivate 
conda activate samapa

#check if package is already in PYTHONPATH
export PATH=/usr/local/cuda/bin:$PATH

if [[ $PYTHONPATH != *"/home/guests/jorge_padilla/code/Guided_Research/SAMAPA:$PYTHONPATH"* ]]; then
    export PYTHONPATH="/home/guests/jorge_padilla/code/Guided_Research/SAMAPA:$PYTHONPATH"
    export PYTHONPATH="/home/guests/jorge_padilla/code/Guided_Research/SAMAPA/preprocess:$PYTHONPATH"
fi

python -u ./debug.py

