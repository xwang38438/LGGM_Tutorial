# Testing Diffusion Reconstruction Ability

domain='reddit'
echo "===========================Load domain: all_wo_${domain}" "Eval domain: ${domain}==========================="

# Specify the full path for the log file
logfile="reddit_test_adj.txt"

# Redirect all output (stdout and stderr) to the logfile
exec > >(tee -a "$logfile") 2>&1

for file in "/home/allenwang/gad_diff/gad_lggm/outputs/reddit_adj/checkpoints/reddit_adj/"*; do
    file=$(echo "$file" | sed 's/=/\\=/g')
    echo "file: ${file}"
    # Check if the file ends with '.ckpt'
    if [[ "$file" == *".ckpt" ]]; then
        # Run the Python command with the current file
        python3 main.py dataset.name="${domain}" general.name="test_${domain}_adj" model.transition="uniform" general.wandb="online" general.ckpt_path="$file" train.batch_size=16 general.setting='test' 
    fi
done
echo "============================================================================================================"





