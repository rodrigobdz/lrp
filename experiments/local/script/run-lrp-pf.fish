#!/usr/bin/env fish
#
# Run experiments on personal machine (not cluster) sequentially

# The following variable should correspond to the value set in batch_lrp_pf.py
set TOTAL_NUMBER_OF_EXPERIMENTS 16

echo 'Deactivating virtualenv, if active'
deactivate

echo 'Activating venv for fish shell'
source ./venv/bin/activate.fish

echo "Running $TOTAL_NUMBER_OF_EXPERIMENTS experiments"
for i in (seq 0 "$TOTAL_NUMBER_OF_EXPERIMENTS")
    echo "Running experiment $i"
    python3 experiments/cluster/script/batch_lrp_pf.py --experiment-id "$i"
end
