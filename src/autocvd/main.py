"""autocvd, a tool for setting CUDA_VISIBLE_DEVICES based on utilization."""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

from autocvd.nvidia_smi_calls import (
    get_free_gpu_memory,
    get_installed_gpus,
    gpu_is_free,
)
from autocvd.utils import Spinner, positive_int

logger = logging.getLogger(__name__)


def autocvd(
    num_gpus: int = 1,
    least_used: bool = False,
    timeout: Optional[int] = None,
    interval: int = 30,
    set_env: bool = True,
    progress: bool = True,
) -> List[int]:
    """Select GPUs based on their utilization.

    Args:
        num_gpus (int, optional): Number of GPUs. Defaults to 1.
        least_used (bool, optional): If True, select least-used GPUs instead of waiting
            for free GPUs. Defaults to False.
        timeout (Optional[int], optional): Timeout for waiting in seconds. Defaults to
            None (=wait indefinitely).
        interval (int, optional): Interval to query GPUs in seconds. Defaults to 30.
        set_env (bool, optional): If True, set CUDA environment variables according to
        selected GPUs. Defaults to True.
        progress (bool, optional): If True, show progress while waiting. Defaults to
            True.

    Raises
    ------
        OSError: If no GPUs are installed.
        TimeoutError: If GPUs could not be acquired before timeout.

    Returns
    -------
        List[int]: Selected GPUs.
    """
    num_installed_gpus = get_installed_gpus()
    if num_installed_gpus == 0:
        raise OSError("There are no GPUs installed.")

    # adjust num_gpus if necessary
    if num_gpus < 1 or num_gpus > num_installed_gpus:
        num_gpus = 1 if num_gpus < 1 else num_installed_gpus
        logger.warning(
            f"Parameter 'num_gpus' must be between 1 and {num_installed_gpus}, setting"
            f" to {num_gpus}."
        )

    # selection
    logger.info(
        f"Selecting {num_gpus} {'least-used' if least_used else 'free'} GPU(s)."
    )
    if least_used:
        free_memories = {
            gpu: get_free_gpu_memory(gpu) for gpu in range(num_installed_gpus)[::-1]
        }
        free_memories = dict(
            sorted(free_memories.items(), key=lambda x: x[1], reverse=True)
        )
        available_gpus = list(free_memories.keys())
    else:
        spinner = Spinner()
        start = time.time()
        while True:
            free_gpus = list(filter(gpu_is_free, range(num_installed_gpus)[::-1]))
            if len(free_gpus) >= num_gpus:
                available_gpus = free_gpus
                break
            for _ in range(interval):
                time_passed = int(time.time() - start)
                if timeout and time_passed > timeout:
                    raise TimeoutError(
                        f"Could not acquire {num_gpus} GPU(s) before timeout."
                    )
                if progress:
                    time_str = (
                        f"for {timeout - time_passed: >{len(str(timeout))}}s"
                        if timeout is not None
                        else "indefinitely"
                    )
                    print(
                        (
                            f"{len(free_gpus)} / {num_gpus} GPU(s) available (waiting"
                            f" {time_str}, querying every {interval}s) {spinner}"
                        ),
                        file=sys.stderr,
                        end="\r",
                    )
                time.sleep(1)
    selected_gpus = sorted(available_gpus[:num_gpus])
    id_str = ",".join(map(str, selected_gpus))
    logger.info(f"Selected GPU(s): {id_str.replace(',', ', ')}.")

    # setting environment variables
    if set_env:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = id_str
        logger.info(
            "Set environment variables"
            f" CUDA_VISIBLE_DEVICES={id_str} and"
            " CUDA_DEVICE_ORDER=PCI_BUS_ID."
        )

    return selected_gpus


def cli() -> None:
    """Command-line interface for autocvd."""
    parser = argparse.ArgumentParser(
        description=(
            "A tool for setting CUDA_VISIBLE_DEVICES based on utilization. Basic usage:"
            " eval $(autocvd) <command>"
        ),
        epilog="Documentation and examples: https://github.com/jonasricker/autocvd",
    )
    parser.add_argument(
        "-n",
        "--num-gpus",
        type=positive_int,
        default=1,
        help="Number of required GPUs. Defaults to 1.",
    )
    parser.add_argument(
        "-l",
        "--least-used",
        action="store_true",
        help=(
            "Select least-used GPUs instead of waiting for free GPUs. Defaults to"
            " False."
        ),
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=positive_int,
        help="Timeout for waiting in seconds. Defaults to no timeout.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=positive_int,
        default=30,
        help="Interval to query GPU status in seconds. Defaults to 30.",
    )
    parser.add_argument(
        "-e",
        "--export",
        action="store_true",
        help="Add 'export' statements such that environment can be sourced.",
    )
    parser.add_argument(
        "-o",
        "--id-only",
        action="store_true",
        help=(
            "Return comma-separated GPU IDs only instead of environment variable"
            " assignment."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print any messages. Defaults to False.",
    )
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(
            stream=sys.stderr, level=logging.INFO, format="autocvd: %(message)s"
        )

    gpus = autocvd(
        num_gpus=args.num_gpus,
        least_used=args.least_used,
        timeout=args.timeout,
        interval=args.interval,
        set_env=False,
        progress=not args.quiet,
    )
    id_str = ",".join(map(str, gpus))
    if args.id_only:
        print(id_str)
    else:
        prefix = "export " if args.export else ""
        print(
            f"{prefix}CUDA_DEVICE_ORDER=PCI_BUS_ID "
            f"{prefix}CUDA_VISIBLE_DEVICES={id_str}"
        )
