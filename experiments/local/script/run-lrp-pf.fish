#!/usr/bin/env fish
#
# Run experiments on personal machine (not cluster) in parallel.

# Check that arguments were passed.
if test -z "$argv[1]"
    echo "Please pass path to configuraton file as first argument.
Usage: ./run_experiments.sh <config_file>"
    exit 1
end

# Export variable to parallel's subshell.
set --export CONFIG_FILE_PATH $argv[1]

# Value is read from config file.
set TOTAL_NUMBER_OF_EXPERIMENTS (grep 'TOTAL_NUMBER_OF_EXPERIMENTS =' ./experiments/local/local.config | awk '{print $3}')

echo "Running $TOTAL_NUMBER_OF_EXPERIMENTS experiments"

# Execute for range of experiments (0 to TOTAL_NUMBER_OF_EXPERIMENTS - 1)
parallel --jobs 50% --line-buffer --tag --halt-on-error now,fail=1 'begin
    function run_experiment --argument-names experiment_id
        echo "Running experiment ID $experiment_id"
        time ./venv/bin/python3 ./experiments/script/batch_lrp_pf.py --experiment-id "$experiment_id" --config-file "$CONFIG_FILE_PATH"

        # Check status of last experiment and exit on error
        if test $status -ne 0
            echo "ERORR: Experiment ID $experiment_id failed"
            exit 1
        end
    end
end
run_experiment {}' ::: (seq 0 (math "$TOTAL_NUMBER_OF_EXPERIMENTS" - 1))

# Exit on error
# Similar to bash set -o errexit, but for fish.
# https://stackoverflow.com/a/19883639
and echo
and echo "Generate plots from experiment results"
and time ./venv/bin/python3 ./experiments/script/visualize.py --config-file "$CONFIG_FILE_PATH"
