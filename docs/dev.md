# Development

How-to guide to build and test `lrp` and `pf` releases locally.

## Requirements

- Docker version `20.10.14`, build `a224086`

## Usage

### Build

```sh
./script/build
```

### Test

Commands to test build locally:

```sh
# Start Docker container as test environment
docker run --interactive --rm --tty --volume lrp/dist:/root/dist python:latest

# Upgrade pip3
python3 -m pip install --upgrade pip

# Install release
python3 -m pip install /root/dist/*.whl

# Test release
python3
import lrp
import pf

# List contents of built packages
help(lrp)
help(pf)
```
