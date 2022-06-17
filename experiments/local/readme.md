# Experiments - Local Environment

Documentation of how to set up and run experiments **locally**.

## Requirements

All commands listed in this guide should be run **locally** (not cluster) from the root directory of this repo.

## Installation

1. Activate virtual environment created in [script/bootstrap](https://github.com/rodrigobdz/lrp/blob/3a99a5461031a18462332247f236cffd81b126b6/script/bootstrap#L10)

   ```sh
   source ./venv/bin/activate
   ```

1. Build project

   ```sh
   ./script/build
   ```

1. Install packaged project

   ```sh
   ./script/install
   ```

## Usage

1. Update the paths in `./experiments/local/local.config`

1. Run experiments and generate plots from results

   ```sh
   # Run multiple batches of experiments
   fish ./experiments/local/script/run-lrp-pf.fish ./experiments/local/local.config
   ```

## Credits

- Scripts follow [rodrigobdz's Shell Style Guide](https://github.com/rodrigobdz/styleguide-sh)
- The structure of this README is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme).
