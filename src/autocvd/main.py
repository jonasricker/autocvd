import argparse
import os
import time
from typing import List, Optional

from autocvd.nvidia_smi_calls import (
    get_free_gpu_memory,
    get_installed_gpus,
    gpu_is_free,
)


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
    # validation
    installed_gpus = get_installed_gpus()
    if num_gpus < 1:
        raise ValueError("Parameter 'num_gpus' must be greater than 0.")
    if num_gpus > len(installed_gpus):
        raise ValueError(
            f"You requested {num_gpus} GPUs, but only"
            f" {len(installed_gpus)} are installed."
        )
    if timeout and timeout < 0:
        raise ValueError("Parameter 'timeout' must be a positive integer.")
    if interval < 0:
        raise ValueError("Parameter 'interval' must be a positive integer.")

    # selection
    if verbose:
        print(
            "Selecting"
            f" {num_gpus} {'least-used' if least_used else 'free'} GPU(s)"
        )

    if least_used:
        free_memories = list(map(get_free_gpu_memory, installed_gpus))
        selected_gpus = [
            gpu
            for _, gpu in sorted(
                zip(free_memories, installed_gpus), reverse=True
            )
        ][:num_gpus]
    else:
        spinner, spinner_idx = "/-\\|", 0
        start = time.time()
        while True:
            free_gpus = list(filter(gpu_is_free, installed_gpus))
            if len(free_gpus) >= num_gpus:
                selected_gpus = free_gpus[-num_gpus:]
                break
            for _ in range(interval):
                if timeout and time.time() - start > timeout:
                    raise TimeoutError(
                        f"Could not acquire {num_gpus} GPU(s) before timeout."
                    )
                if verbose:
                    print(
                        f"Waiting for {num_gpus} free GPU(s) (Timeout:"
                        f" {str(timeout) + 's' if timeout else 'inf'})"
                        f" {spinner[spinner_idx]}",
                        end="\r",
                    )
                    spinner_idx = (spinner_idx + 1) % len(spinner)
                time.sleep(1)
    if verbose:
        print(f"Selected GPU(s): {selected_gpus}.")

    # setting environment variables
    if set_env:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

    return selected_gpus


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select GPUs based on their utilization. To set the environment"
            " variables CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER for a"
            " single command run 'eval $(autocvd <args>) <command>'. To source"
            " them into the current shell environment run '. <(autocvd -s"
            " <args>)'."
        ),
        epilog="Documentation: https://github.com/jonasricker/autocvd",
    )
    parser.add_argument(
        "-n",
        "--num-gpus",
        type=int,
        default=1,
        help="Number of required GPUs. Defaults to 1.",
    )
    parser.add_argument(
        "-l",
        "--least-used",
        action="store_true",
        help=(
            "Instead of waiting for free GPUs, use least used. Defaults to"
            " False."
        ),
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        help="Timeout for waiting in seconds. Defaults to None.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=30,
        help="Interval to query GPU status in seconds. Defaults to 30.",
    )
    parser.add_argument(
        "-s",
        "--source",
        action="store_true",
        help=(
            "Add 'export' statements to output such that environment can be"
            " sourced."
        ),
    )
    args = parser.parse_args()

    try:
        gpus = autocvd(
            num_gpus=args.num_gpus,
            least_used=args.least_used,
            timeout=args.timeout,
            interval=args.interval,
            set_env=False,
            verbose=False,
        )
        print(f"echo Selected GPU\(s\): {gpus};")  # noqa
        prefix = "export " if args.source else ""
        print(
            f"{prefix}CUDA_DEVICE_ORDER=PCI_BUS_ID "
            f"{prefix}CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}"
        )
    except (ValueError, TimeoutError) as e:
        print(f"echo '{e}'")
