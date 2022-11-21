import subprocess
from typing import List


def get_installed_gpus() -> List[int]:
    output = subprocess.check_output(["nvidia-smi", "-L"]).decode()
    num_gpus = len(output.strip().split("\n"))
    return list(range(num_gpus))


def gpu_is_free(gpu: int) -> bool:
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
