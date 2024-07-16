domain='reddit_onehot'
echo "===========================Load domain: all_wo_${domain}" "Eval domain: ${domain}==========================="

# Specify the full path for the log file
logfile="terminal_outputs/reddit_augmentation.txt"

# Redirect all output (stdout and stderr) to the logfile
exec > >(tee -a "$logfile") 2>&1

for file in "/home/allenwang/gad_diff/gad_lggm/outputs/reddit_onehot/checkpoints/reddit_onehot/"*; do
    file=$(echo "$file" | sed 's/=/\\=/g')
    echo "file: ${file}"
    # Check if the file ends with '.ckpt'
    if [[ "$file" == *".ckpt" ]]; then
        # Run the Python command with the current file
        python3 main.py dataset.name="${domain}" general.name="aug_${domain}" model.transition="uniform" general.wandb="disabled" general.ckpt_path="$file" train.batch_size=12 general.setting='augment' 
    fi
done
echo "============================================================================================================"

