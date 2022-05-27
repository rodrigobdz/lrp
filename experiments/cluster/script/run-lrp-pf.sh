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
#$ -t 0-15            # start multiple instances with identified SGE_TASK_ID from 0 to n
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
# qsub -m esa -M r.bermudezschettino@campus.tu-berlin.de run-lrp-pf.sh

# Bash options
set -o errexit  # exit on error
set -o pipefail # exit on pipe failure
set -o nounset  # exit on unset variable

case "$SGE_TASK_ID" in
# Verify cluster job IDs
0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15) ;;
# Error handling for unsupported experiment IDs
*)
  echo "Unsupported SGE_TASK_ID $SGE_TASK_ID. Exiting..." && exit 1
  ;;
esac

# Log environment
echo "SGE_TASK_ID: $SGE_TASK_ID"

python3 ./experiments/script/batch_lrp_pf.py --experiment-id "$SGE_TASK_ID"
