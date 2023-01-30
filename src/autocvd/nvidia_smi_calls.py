"""Calls to nvidia-smi."""

import subprocess


def get_installed_gpus() -> int:
    """Return number of installed GPUs."""
    output = subprocess.check_output(["nvidia-smi", "-L"]).decode()
    num_gpus = len(output.strip().split("\n"))
    return num_gpus


def gpu_is_free(gpu: int) -> bool:
    """Test if GPU is free (not used by any compute jobs)."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu),
        ]
    ).decode()
    return output == ""


def get_free_gpu_memory(gpu: int) -> int:
    """Return free memory on GPU."""
    output = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ]
        )
        .decode()
        .strip()
    )
    return int(output)
