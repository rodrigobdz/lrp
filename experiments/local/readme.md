# Experiments - Local Environment

Documentation of how to set up and run experiments **locally**.

## Requirements

All commands listed in this guide should be run **locally** (not cluster) from the root directory of this repo.

## Installation

1. Build project

   ```sh
   ./script/build
   ```

1. Install packaged project

   ```sh
   ./script/install
   ```

## Usage

1. Run experiments

   ```sh
   # Run multiple batches of experiments
   fish ./experiments/local/script/run-lrp-pf.fish
   ```

1. Visualize results

   ```sh
   python3 ./experiments/script/visualize.py
   ```

## Credits

- Scripts follow [rodrigobdz's Shell Style Guide](https://github.com/rodrigobdz/styleguide-sh)
- The structure of this README is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme).
