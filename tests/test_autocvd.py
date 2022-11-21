import os
from typing import List

import pytest
from pytest_mock import MockerFixture

from autocvd.main import autocvd


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
        (3, 3, [0, 1, 2], [0, 1, 2]),
    ],
)
def test_select_free(
    num_installed: int,
    num_requested: int,
    free_gpus: List[int],
    expected: List[int],
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "autocvd.main.get_installed_gpus",
        return_value=list(range(num_installed)),
    )
    mocker.patch(
        "autocvd.main.gpu_is_free", side_effect=lambda gpu: gpu in free_gpus
    )

    assert autocvd(num_gpus=num_requested) == expected


@pytest.mark.parametrize(
    "num_installed,num_requested,memories,expected",
    [
        (1, 1, [100], [0]),
        (2, 1, [100, 99], [0]),
        (2, 1, [99, 100], [1]),
        (2, 2, [100, 99], [0, 1]),
        (2, 2, [99, 100], [1, 0]),
        (3, 1, [100, 99, 98], [0]),
        (3, 1, [99, 100, 99], [1]),
        (3, 1, [98, 99, 100], [2]),
        (3, 2, [100, 99, 98], [0, 1]),
        (3, 2, [99, 100, 98], [1, 0]),
        (3, 3, [98, 99, 100], [2, 1, 0]),
    ],
)
def test_select_least_used(
    num_installed: int,
    num_requested: int,
    memories: List[int],
    expected: List[int],
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "autocvd.main.get_installed_gpus",
        return_value=list(range(num_installed)),
    )
    mocker.patch(
        "autocvd.main.get_free_gpu_memory",
        side_effect=lambda gpu: memories[gpu],
    )

    assert autocvd(num_gpus=num_requested, least_used=True) == expected


@pytest.mark.parametrize(
    "num_gpus,timeout,interval,raises",
    [
        (1, None, 30, False),
        (1, 1, 1, False),
        (0, None, 1, True),
        (1, -1, 1, True),
        (1, 1, -1, True),
    ],
)
def test_parameter_validation(
    num_gpus: int,
    timeout: int,
    interval: int,
    raises: bool,
    mocker: MockerFixture,
) -> None:
    mocker.patch("autocvd.main.get_installed_gpus", return_value=[1])
    if raises:
        with pytest.raises(ValueError):
            autocvd(num_gpus=num_gpus, timeout=timeout, interval=interval)
    else:
        autocvd(num_gpus=num_gpus, timeout=timeout, interval=interval)


@pytest.mark.parametrize(
    "num_installed,num_requested,raises",
    [(1, 1, False), (1, 2, True), (1, 0, True)],
)
def test_gpu_validation(
    num_installed: int, num_requested: int, raises: bool, mocker: MockerFixture
) -> None:
    mocker.patch(
        "autocvd.main.get_installed_gpus",
        return_value=list(range(num_installed)),
    )
    mocker.patch("autocvd.main.gpu_is_free", return_value=True)
    if raises:
        with pytest.raises(ValueError):
            autocvd(num_gpus=num_requested)
    else:
        autocvd(num_gpus=num_requested)


def test_timeout(mocker: MockerFixture) -> None:
    mocker.patch(
        "autocvd.main.get_installed_gpus",
        return_value=[0],
    )
    mocker.patch("autocvd.main.gpu_is_free", return_value=False)

    with pytest.raises(TimeoutError):
        autocvd(num_gpus=1, timeout=1)


def test_autocvd(mocker: MockerFixture) -> None:
    mocker.patch(
        "autocvd.main.get_installed_gpus",
        return_value=[0],
    )
    mocker.patch("autocvd.main.gpu_is_free", return_value=True)

    autocvd()
    assert (
        os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        and os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    )
