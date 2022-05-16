# Cluster Experiments

Documentation of how to set up and run experiments on a cluster.

All commands listed in this guide should be run on execution nodes on the cluster.

## Requirements

- `./cluster/script/bootstrap-ubuntu` - install dependencies for the first time

## Usage

1. `./cluster/script/setup` - Prepare environment each time before running experiments
2. `bash ./cluster/script/run-lrp-pf.sh` - Run experiments on the cluster

   `fish ./local/script/run-lrp-pf.fish` - Alternatively, run experiments locally

3. `python3 cluster/script/visualize.py` - Visualize results

## Credits

- Scripts follow [Personal Shell Style Guide](https://github.com/rodrigobdz/styleguide-sh)
- The structure of this README is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme).
