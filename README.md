# autocvd - Automatically set `CUDA_VISIBLE_DEVICES`

Are you working on a shared system with multiple NVIDIA GPUs and tired of manually setting `CUDA_VISIBLE_DEVICES` every time you run a program? Then, *autocvd* might be the right tool for you!

Basic usage is as simple as:
```
$ eval $(autocvd) your_awesome_program
```
*autocvd* will identify a free GPU and set `CUDA_VISIBLE_DEVICES` accordingly. If no GPU is free, it will wait until one becomes available. This behavior can be customized (see [Usage](#usage)).

## Features
- no dependencies
- can be used from command line and code
- no code changes required (if used from the command line)

## Requirements
*autocvd* uses [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) to query GPU utilization.
Make sure that it is installed and callable.

## Installation
```bash
pip install autocvd
```
## Usage
### Command Line
```
$ autocvd -h
usage: autocvd [-h] [-n NUM_GPUS] [-l] [-t TIMEOUT] [-i INTERVAL] [-s]

Select GPUs based on their utilization. To set the environment variables CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER for a single command run 'eval $(autocvd <args>) <command>'. To
source them into the current shell environment run '. <(autocvd -s <args>)'.

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_GPUS, --num-gpus NUM_GPUS
                        Number of required GPUs. Defaults to 1.
  -l, --least-used      Instead of waiting for free GPUs, use least used. Defaults to False.
  -t TIMEOUT, --timeout TIMEOUT
                        Timeout for waiting in seconds. Defaults to None.
  -i INTERVAL, --interval INTERVAL
                        Interval to query GPU status in seconds. Defaults to 30.
  -s, --source          Add 'export' statements to output such that environment can be sourced.
```

### Code
```python
from autocvd import autocvd


autocvd()
```

```python
def autocvd(
    num_gpus: int = 1,
    least_used: bool = False,
    timeout: Optional[int] = None,
    interval: int = 30,
    set_env: bool = True,
    verbose: bool = True,
) -> List[int]:
    """Select GPUs based on their utilization.

    Args:
        num_gpus: Number of required GPUs. Defaults to 1.
        least_used: Instead of waiting for free GPUs, use least used.
            Defaults to False.
        timeout: Timeout for waiting in seconds. Defaults to None.
        interval: Interval to query GPU status in seconds. Defaults to 30.
        set_env: Set environment variables according to selected GPUs.
            Defaults to True.
        verbose: Print additional information. Defaults to True.

    Raises:
        ValueError: If arguments are invalid.
        TimeoutError: If GPUs could not be acquired after `timeout` seconds.

    Returns:
        A list containing the selected GPUs.
    """
```

## Notes
- Settings `CUDA_DEVICE_ORDER=PCI_BUS_ID` is required to ensure that the ordering of CUDA devices is the same for `nvidia-smi` and other programs.

## Similar Projects
- [cuthon](https://github.com/awni/cuthon)
- [setGPU](https://github.com/bamos/setGPU)
