#!/usr/bin/env bash
#
# Run Layer-wise Relevance Propagation and Pixel-Flipping experiments on cluster.
#
#   Outputs will be saved to script.sh.o<job id>.<task id>.
#   Source: https://wiki.ml.tu-berlin.de/wiki/IDA/TheTuCluster
#
# Cluster - Univa Grid Engine (UGE) options:
#
#$ -binding linear:4  # request 4 CPUs (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N lrppf           # set job name, used for naming output files
#$ -l cuda=1          # request one GPU
#$ -l h_vmem=16G      # request 16GB of RAM
#$ -l mem_free=16G    # request 16GB of free memory
#$ -q all.q           # submit jobs to queue named all.q (general queue)
#$ -cwd               # execute in current working directory
#$ -t 1-256           # start multiple instances with identified SGE_TASK_ID from 1 to n
#                     # upper bound of -t should be equal to TOTAL_NUMBER_OF_EXPERIMENTS in *.config
#
# Submit jobs to cluster using qsub
#
#   qsub options:
#
#     -m <mail_options>
#       'e' Mail is sent at the end of the job.
#       'a' Mail is sent when the job is aborted or rescheduled.
#       's' Mail is sent when the job is suspended.
#     -M <email address> # send mail to this address on the events specified
#
# qsub -m esa -M r.bermudezschettino@campus.tu-berlin.de /home/rodrigo/experiments/cluster/script/run-lrp-pf.sh

# Bash options
set -o errexit  # exit on error
set -o pipefail # exit on pipe failure
set -o nounset  # exit on unset variable

# Subtract one because SGE_TASK_ID can only start at 1 and not 0.
EXPERIMENT_ID="$((SGE_TASK_ID - 1))"
readonly EXPERIMENT_ID

# Log environment
echo "SGE_TASK_ID: $SGE_TASK_ID. Argument for Python script EXPERIMENT_ID: $EXPERIMENT_ID"

# Disable shellcheck warning about using source with a variable.
# shellcheck disable=SC1091
source /home/rodrigo/experiments/cluster/script/setup

# Run experiments
time python3 /home/rodrigo/experiments/script/batch_lrp_pf.py --experiment-id "$EXPERIMENT_ID" --config-file /home/rodrigo/experiments/cluster/cluster.config

# After last experiment, generate plots from aggregated results.
if [ "$SGE_TASK_ID" -eq 256 ]; then
  echo
  echo "Generate plots from experiment results"
  time python3 /home/rodrigo/experiments/script/visualize.py --config-file /home/rodrigo/experiments/cluster/cluster.config
fi
