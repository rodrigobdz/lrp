# Cluster Experiments

Documentation of how to set up and run experiments on the **cluster**.

## Requirements

All commands listed in this guide should be run on execution nodes on the **cluster**.

## Installation

1. Build project

   ```sh
   # Locally
   ./script/build
   ```

1. Transfer project to execution nodes

   ```sh
   # Locally
   ./experiments/cluster/script/transfer-build-to-cluster
   ```

1. Log in to cluster

   ```sh
   # Assuming you have set up ~/.ssh/config with an entry for the cluster with HostName 'ml'
   ssh ml
   ```

1. Install dependencies on cluster

   ```sh
   # On Cluster
   cd /home/rodrigo/
   ./experiments/cluster/script/bootstrap-cluster
   ```

## Usage

1. Log in to cluster

1. Update the paths in `/home/rodrigo/experiments/server/server.config`

1. Install code for experiments

   ```sh
   # On Cluster

   # Source setup script to load conda environment, if installed
   source /home/rodrigo/experiments/cluster/script/setup

   # qlogin not needed if jobs are not going to be submitted interactively
   /home/rodrigo/experiments/cluster/script/install
   ```

1. Run experiments

   Non-interactive jobs:

   ```sh
   # Make script executable
   chmod +x /home/rodrigo/experiments/cluster/script/submit-cluster-jobs

   # Run experiments
   /home/rodrigo/experiments/cluster/script/submit-cluster-jobs
   ```

   Interactive jobs:

   ```sh
   # Log in to node with CUDA support.
   qlogin -l cuda=1

   # Source setup script to load conda environment, if installed
   source /home/rodrigo/experiments/cluster/script/setup

   # Run 16 experiments sequentially
   for i in {0..15}; do
      python3 ./experiments/script/batch_lrp_pf.py --experiment-id $i --config-file ./experiments/cluster/cluster.config
   done
   ```

1. Generate plots from results and download from cluster

   ```sh
   # Locally
   ./experiments/cluster/script/download-results-from-cluster
   ```

## One-Liners for Copy-Pasting

Update build on cluster:

```sh
# Locally
./script/build && ./experiments/cluster/script/transfer-build-to-cluster && ssh ml 'source /home/rodrigo/experiments/cluster/script/setup && /home/rodrigo/experiments/cluster/script/install' && ssh ml

# cluster

# non-interactive
chmod +x /home/rodrigo/experiments/cluster/script/submit-cluster-jobs && /home/rodrigo/experiments/cluster/script/submit-cluster-jobs

# interactive
qlogin -l cuda=1
source /home/rodrigo/experiments/cluster/script/setup && python3 ./experiments/script/batch_lrp_pf.py --experiment-id 0 --config-file ./experiments/cluster/cluster.config
```

## Credits

- Scripts follow [rodrigobdz's Shell Style Guide](https://github.com/rodrigobdz/styleguide-sh)
- The structure of this README is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme).
