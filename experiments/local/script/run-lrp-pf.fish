#!/usr/bin/env fish
#
# Run experiments on personal machine (not cluster) sequentially.


# Check that arguments were passed.
if test -z "$argv[1]"
    echo "Please pass path to configuraton file as first argument.
Usage: ./run_experiments.sh <config_file>"
    exit 1
end

# The following variable should correspond to the value set in batch_lrp_pf.py
set TOTAL_NUMBER_OF_EXPERIMENTS 16

echo "Running $TOTAL_NUMBER_OF_EXPERIMENTS experiments"
for i in (seq 0 (math "$TOTAL_NUMBER_OF_EXPERIMENTS" - 1))
    echo "Running experiment ID $i"
    ./venv/bin/python3 ./experiments/script/batch_lrp_pf.py --experiment-id "$i" --config-file "$argv[1]"
end
