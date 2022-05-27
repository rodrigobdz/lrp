# Cluster Experiments

Documentation of how to set up and run experiments on a cluster.

All commands listed in this guide should be run on execution nodes on the cluster.

## Requirements

- Install dependencies for the first time

  ```sh
  ./cluster/script/bootstrap-ubuntu
  ```

## Usage

1. Prepare environment each time before running experiments

   ```sh
   ./cluster/script/setup
   ```

2. Run experiments on the **cluster**

   ```sh
   bash ./cluster/script/run-lrp-pf.sh
   ```

   Alternatively, run experiments **locally**

   ```sh
   # Run from root directory of this (lrp) repo
   fish ./experiments/local/script/run-lrp-pf.fish
   ```

3. Visualize results

   ```sh
   python3 ./experiments/script/visualize.py
   ```

## Credits

- Scripts follow [Personal Shell Style Guide](https://github.com/rodrigobdz/styleguide-sh)
- The structure of this README is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme).
