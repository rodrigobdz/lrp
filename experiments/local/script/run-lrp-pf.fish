#!/usr/bin/env fish
#
# Run experiments on personal machine (not cluster) sequentially.


# Check that arguments were passed.
if test -z "$argv[1]"
    echo "Please pass path to configuraton file as first argument.
Usage: ./run_experiments.sh <config_file>"
    exit 1
end

# Value is read from config file.
set TOTAL_NUMBER_OF_EXPERIMENTS (grep 'TOTAL_NUMBER_OF_EXPERIMENTS =' ./experiments/local/local.config | awk '{print $3}')

echo "Running $TOTAL_NUMBER_OF_EXPERIMENTS experiments"

# Execute for range of experiments (0 to TOTAL_NUMBER_OF_EXPERIMENTS - 1)
for i in (seq 0 (math "$TOTAL_NUMBER_OF_EXPERIMENTS" - 1))
    echo "Running experiment ID $i"
    time ./venv/bin/python3 ./experiments/script/batch_lrp_pf.py --experiment-id "$i" --config-file "$argv[1]"

    # Check status of last experiment and exit on error
    if test $status -ne 0
        echo "ERORR: Experiment ID $i failed"
        exit 1
    end
end
