#!/bin/bash
# For hyperparameter tuning with grid search only. If you would like to run the reported main experiments, use:
    # python main.py <job_name>
# where <job_name> corresponds to the folder under emforecaster/jobs/exp/. For example, python main.py open_neuro/finite-context/patchtst

# ps aux | grep "[p]ython main.py" | awk '{print $2}' | xargs kill -9
export CUDA_LAUNCH_BLOCKING=1

source /home/fam/xmootoo/torch2024_new/bin/activate

# Define list of ablations
ablations=(
    # "open_neuro/finite-context/dlinear"
    # "open_neuro/finite-context/timesnet"
    # "open_neuro/finite-context/patchtst"
    # "open_neuro/finite-context/moderntcn"
    "open_neuro/infinite-context/dtw"
)

# Set this to true for sequential job execution, false for parallel
sequential=true

# Function to run a single ablation
run_ablation() {
    local ablation=$1
    echo "Running ablation: $ablation"
    python emforecaster/tuning/tune.py "$ablation"
}

# Main execution logic
if [ "$sequential" = true ]; then
    echo "Running ablations sequentially"
    for ablation in "${ablations[@]}"; do
        run_ablation "$ablation"
    done
else
    echo "Running ablations in parallel"
    for ablation in "${ablations[@]}"; do
        run_ablation "$ablation" &
    done
    # Wait for all background processes to finish
    wait
fi

echo "All ablations completed"
