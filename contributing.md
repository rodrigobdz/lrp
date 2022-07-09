# Welcome to our contributing guide

The following sections will help you set up your development environment and get started on your contribution.

## Requirements

- `pre-commit`>=2.19.0

## Installation

Install dependencies

```sh
sudo ./script/bootstrap
```

## Usage

Refer to [lrp.ipynb](./lrp.ipynb) for an example.

### Lint

Lint all files manually:

```sh
 pre-commit run --all-files
```

### Uploading to PyPI

1. Configure PyPI credentials in `$HOME/.pypirc`

1. Build and upload package

   ```sh
   ./script/build
   ./script/upload
   ```

Full documentation [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## Credits

- The structure of this readme is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme)
