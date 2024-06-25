from typing import Any


def remove_null(data: list[Any]) -> list[Any]:
    return [x for x in data if x]


def mean(data: list[float]) -> float:
    """mean(data: list[float]) -> float
    return the mean of data
    """
    data = remove_null(data)
    return sum(data) / len(data)


def median(data: list[float]) -> float | int:
    """median(data: list[float]) -> float
    return the median of data
    """
    data = remove_null(data)
    if len(data) % 2 != 0:
        return data[len(data) // 2]
    first_med = data[len(data) // 2]
    second_med = data[(len(data) // 2) + 1]
    return (data[first_med] + data[second_med]) // 2
