#!/usr/bin/env fish
#
# Run experiments on personal machine (not cluster) sequentially.

# The following variable should correspond to the value set in batch_lrp_pf.py
set TOTAL_NUMBER_OF_EXPERIMENTS 16

echo "Running $TOTAL_NUMBER_OF_EXPERIMENTS experiments"
for i in (seq 0 (math "$TOTAL_NUMBER_OF_EXPERIMENTS" - 1))
    echo "Running experiment ID $i"
    python3 ./experiments/script/batch_lrp_pf.py --experiment-id "$i"
end
