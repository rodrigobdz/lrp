# Cluster Experiments

Documentation of how to set up and run experiments on a cluster.

All commands listed in this guide should be run on execution nodes on the cluster.

## Installation

1. Build project

   ```sh
   ./script/build
   ```

1. Install dependencies on local machine

   ```sh
   ./experiments/cluster/script/bootstrap-local
   ```

1. Install dependencies on cluster

   ```sh
   ./experiments/cluster/script/bootstrap-cluster
   ```

## Usage

<!-- markdownlint-disable ol-prefix -->

4. `Cluster only` Prepare environment each time before running experiments

   ```sh
   # Prepare environment to run experiments on cluster
   ./experiments/cluster/script/setup

   # Install code for experiments
   ./script/install
   ```

5. Run experiments

   1. either on the **cluster**:

      ```sh
      bash ./experiments/cluster/script/run-lrp-pf.sh
      ```

   1. or **locally**:

      ```sh
      # Run from root directory of this (lrp) repo
      fish ./experiments/local/script/run-lrp-pf.fish
      ```

6. Visualize results

   ```sh
   python3 ./experiments/script/visualize.py
   ```

<!-- markdownlint-enable ol-prefix -->

## Credits

- Scripts follow [rodrigobdz's Shell Style Guide](https://github.com/rodrigobdz/styleguide-sh)
- The structure of this README is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme).
