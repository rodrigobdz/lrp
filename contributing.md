# Welcome to our contributing guide

The following sections will help you set up your development environment and get started on your contribution.

## Requirements

- `pre-commit`>=2.19.0

Changes have been tested with the following dependency versions:

- `numpy>=1.21.4`
- `matplotlib>=3.5.2`
- `opencv-contrib-python-headless>=4.5.4.60`
- `scikit-learn>=1.0.2`
- `torch>=1.10.0`
- `torchvision>=0.11.1`

## Installation

Install dependencies

```sh
sudo ./script/bootstrap
```

## Usage

Refer to [lrp.ipynb](lrp.ipynb) for an example.

### Lint

Lint all files manually:

```sh
 pre-commit run --all-files
```

### Uploading to PyPI

1. Configure PyPI credentials in `$HOME/.pypirc`

1. Bump version in [setup.cfg](https://github.com/rodrigobdz/lrp/blob/main/setup.cfg#L3)

1. Bump version in [README.md](https://github.com/rodrigobdz/lrp/commit/c27003dd669c3e6a34af1f3e864dbe22a0b562c4)

1. Create GitHub release. [Example](https://github.com/rodrigobdz/lrp/releases/tag/v0.1.5)

1. Update zenodo links in [README.md](https://github.com/rodrigobdz/lrp/commit/d19163f140c80075911ebb2b5234030a312ab4f8)

1. Build and upload package

   ```sh
   ./script/build
   ./script/upload
   ```

Full documentation [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## Credits

- The structure of this readme is based on [minimal-readme](https://github.com/rodrigobdz/minimal-readme)
