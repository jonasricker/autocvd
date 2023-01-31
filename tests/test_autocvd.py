import os
from typing import List

import pytest
from autocvd.main import autocvd
from pytest_mock import MockerFixture


@pytest.fixture
def nvidia_smi_mocker(mocker: MockerFixture):
    mocker.patch("autocvd.main.get_installed_gpus", return_value=1)
    mocker.patch("autocvd.main.gpu_is_free", return_value=True)
    mocker.patch("autocvd.main.get_free_gpu_memory", return_value=1)
    return mocker


@pytest.mark.parametrize(
    "num_installed,num_requested,free_gpus,expected",
    [
        (1, 1, [0], [0]),
        (2, 1, [0], [0]),
        (2, 1, [1], [1]),
        (2, 1, [0, 1], [1]),
        (2, 2, [0, 1], [0, 1]),
        (3, 1, [0], [0]),
        (3, 1, [0, 1], [1]),
        (3, 1, [0, 1, 2], [2]),
        (3, 2, [0, 1], [0, 1]),
        (3, 2, [0, 1, 2], [1, 2]),
        (3, 3, [0, 1, 2], [0, 1, 2]),
    ],
)
def test_select_free(
    num_installed: int,
    num_requested: int,
    free_gpus: List[int],
    expected: List[int],
    nvidia_smi_mocker: MockerFixture,
) -> None:
    nvidia_smi_mocker.patch(
        "autocvd.main.get_installed_gpus",
        return_value=num_installed,
    )
    nvidia_smi_mocker.patch(
        "autocvd.main.gpu_is_free", side_effect=lambda gpu: gpu in free_gpus
    )

    assert autocvd(num_gpus=num_requested) == expected


@pytest.mark.parametrize(
    "num_installed,num_requested,memories,expected",
    [
        (1, 1, [1], [0]),
        (2, 1, [2, 1], [0]),
        (2, 1, [1, 2], [1]),
        (2, 2, [2, 1], [0, 1]),
        (2, 1, [1, 1], [1]),
        (3, 1, [3, 2, 1], [0]),
        (3, 1, [1, 2, 1], [1]),
        (3, 1, [1, 1, 1], [2]),
        (3, 2, [3, 2, 1], [0, 1]),
        (3, 2, [1, 1, 1], [1, 2]),
        (3, 3, [1, 2, 3], [0, 1, 2]),
    ],
)
def test_select_least_used(
    num_installed: int,
    num_requested: int,
    memories: List[int],
    expected: List[int],
    nvidia_smi_mocker: MockerFixture,
) -> None:
    nvidia_smi_mocker.patch(
        "autocvd.main.get_installed_gpus",
        return_value=num_installed,
    )
    nvidia_smi_mocker.patch(
        "autocvd.main.get_free_gpu_memory",
        side_effect=lambda gpu: memories[gpu],
    )

    assert autocvd(num_gpus=num_requested, least_used=True) == expected


@pytest.mark.parametrize(
    "num_installed,num_requested,expected",
    [(1, 0, [0]), (1, 2, [0]), (2, 3, [0, 1])],
)
def test_num_gpu_validation(
    num_installed: int,
    num_requested: int,
    expected: List[int],
    nvidia_smi_mocker: MockerFixture,
) -> None:
    nvidia_smi_mocker.patch(
        "autocvd.main.get_installed_gpus",
        return_value=num_installed,
    )
    assert autocvd(num_gpus=num_requested) == expected


def test_env(nvidia_smi_mocker: MockerFixture) -> None:
    autocvd()
    assert (
        os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        and os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    )


def test_timeout(nvidia_smi_mocker: MockerFixture) -> None:
    nvidia_smi_mocker.patch("autocvd.main.gpu_is_free", return_value=False)

    with pytest.raises(TimeoutError):
        autocvd(num_gpus=1, timeout=1)


def test_no_gpus(nvidia_smi_mocker: MockerFixture) -> None:
    nvidia_smi_mocker.patch("autocvd.main.get_installed_gpus", return_value=0)

    with pytest.raises(OSError):
        autocvd()
