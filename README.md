# autocvd, a tool for setting CUDA_VISIBLE_DEVICES based on utilization

On a system with multiple NVIDIA GPUs, *autocvd* **eliminates the need for manually specifying the `CUDA_VISIBLE_DEVICES` environment variable**. This comes in especially handy on systems with multiple users, like a **shared GPU server**. It is **dependency-free** and requires **no code changes** in your scripts.

To execute a command on a single free GPU, run
```bash
$ eval $(autocvd) <command>
```
This will select a free GPU (or wait if none is available) and run the command with the appropriate environment variables set.

For ease of use you might want to define an alias in your `.bashrc`, e.g., to run a Python script on a free GPU:
```bash
$ alias gpupython="eval $(autocvd) python"
```

## Examples
```bash
# run command on two free GPUs
$ eval $(autocvd -n 2) <command>

# run command on least-used GPU (i.e., do not wait if no GPU is free)
$ eval $(autocvd -l) <command>

# exclude certain GPUs
$ eval $(autocvd -x 0 2) <command>

# if no free GPU is available immediately, wait for 60 seconds only
$ eval $(autocvd -t 60) <command>

# export environment variables into the current shell
$ . <(autocvd -e)  # alternatively: source <(autocvd -e)
```

## Requirements
*autocvd* uses [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) to query GPU utilization.
Make sure that it is installed and callable.

## Installation
```bash
pip install autocvd
```
## Usage
```
usage: autocvd [-h] [-n NUM_GPUS] [-l] [-t TIMEOUT] [-i INTERVAL] [-e] [-o] [-q]

A tool for setting CUDA_VISIBLE_DEVICES based on utilization. Basic usage: eval $(autocvd) <command>

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_GPUS, --num-gpus NUM_GPUS
                        Number of required GPUs. Defaults to 1.
  -l, --least-used      Select least-used GPUs instead of waiting for free GPUs. Defaults to False.
  -x EXCLUDE [EXCLUDE ...], --exclude EXCLUDE [EXCLUDE ...]
                        One or multiple GPUs (separated by space) to be excluded. Defaults to no GPU being excluded.
  -t TIMEOUT, --timeout TIMEOUT
                        Timeout for waiting in seconds. Defaults to no timeout.
  -i INTERVAL, --interval INTERVAL
                        Interval to query GPU status in seconds. Defaults to 30.
  -e, --export          Add 'export' statements such that environment can be sourced.
  -o, --id-only         Return comma-separated GPU IDs only instead of environment variable assignment.
  -q, --quiet           Do not print any messages. Defaults to False.
```

*autocvd* can also be used to set the environment variables from a Python script itself:
```python
from autocvd import autocvd


autocvd()

# code accessing GPUs
```
Note that some packages read `CUDA_VISIBLE_DEVICES` when being imported, which makes it necessary to call *autocvd* **before** importing it.


## Notes
- Besides setting `CUDA_VISIBLE_DEVICES`, *autocvd* also sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`. This is required to ensure that the ordering of CUDA devices is consistent.

## Related Projects
- [cuthon](https://github.com/awni/cuthon)
- [setGPU](https://github.com/bamos/setGPU)
