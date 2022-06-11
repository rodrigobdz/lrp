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

   ```sh
   # Change directory to log to save results in this directory
   cd /home/rodrigo/log

   # Submit experiment jobs
   qsub -m esa -M r.bermudezschettino@campus.tu-berlin.de /home/rodrigo/experiments/cluster/script/run-lrp-pf.sh
   ```

1. Visualize results

   ```sh
   # On Cluster
   cd /home/rodrigo
   python3 ./experiments/script/visualize.py --config-file ./experiments/cluster/cluster.config
   ```

## Credits

- Scripts follow [rodrigobdz's Shell Style Guide](https://github.com/rodrigobdz/styleguide-sh)
- The structure of this README is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme).
