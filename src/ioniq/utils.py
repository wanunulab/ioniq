#!/usr/bin/env python
"""
Utility functions and classes for internal and external use
"""
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import scipy.signal as signal
from typing import Literal
from ioniq.core import MetaSegment

# try:
#     import cupy
#     import cupyx.scipy.signal as signal
#     if not cupy.cuda.is_available():
#         raise ImportError

#     np = cupy
# except ImportError:

#     import numpy as np
#     from scipy import signal

class Singleton(type):
    """
    Generic singleton metaclass.

    This metaclass ensures the same return every time the class is instanced.
    """

    def __init__(self, *args, **kwargs):
        """
        Class initialization

        :param args: Arguments for the class
        :param kwargs: Keyword arguments for the class
        """
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Returns the Singleton instance.

        It creates the instance if it doesn't exist the instance does not exist.
        If the instance exists, it is returned as-is.

        :param args: Arguments for the class constructor.
        :param kwargs: Keyword arguments for the class constructor.
        :return: The Singleton instance.
        :rtype: object
        """
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
        return self.__instance


def split_voltage_steps(voltage: np.ndarray, n_remove=0, as_tuples=False):
    """
    Split the voltage signal into steps.

    This function splits the signal at the points where the voltage changes.

    :param voltage: The voltage signal.
    :type voltage: numpy.ndarray
    :param n_remove: Number of points to remove from the start of each split, defaults to 0.
    :type n_remove: int
    :param as_tuples: Return the splits as tuples of start and end indices, defaults=False.
    :type as_tuples: bool
    :raises ValueError: If `n_remove` is negative or larger than the start of the voltage changes.
    :return: Start and end indices of the splits.
    :rtype: tuple of numpy.ndarray or list of tuple if as_tuples=True
    """
    # Check if the current or voltage arrays are empty
    if not voltage.size:
        return []
    # Check if the n_remove argument is negative
    if n_remove < 0:
        raise ValueError("n_remove must be non-negative")
    # Find the indices at which the voltage level changes
    split_indices = np.where(voltage[:-1] != voltage[1:])[0] + 1
    # Add the start and end indices of the current array to the split indices
    split_indices = np.concatenate([[0], split_indices, [len(voltage)]])
    # Calculate the start and end indices of the splits
    start_indices = split_indices[:-1] + n_remove
    end_indices = split_indices[1:]
    # Check if n_remove is not too large
    if np.any(end_indices <= start_indices):
        raise ValueError("n_remove is too large")
    if not as_tuples:
        return start_indices, end_indices
    # Optional: return list of tuples of start and end
    return [(start_ind, end_ind) for start_ind, end_ind in zip(start_indices, end_indices)]


def si_eval(value, unit=None, return_unit=False):
    """
    Convert a value with SI prefix to its equivalent value.

    If the `value` is a string, it splits the numeric value and the unit and
    applies SI prefix multiplier.

    If the value is already numeric, it directly applies SI prefix multiplier.

    :param value: The value to be converted.
    :type value: str or float
    :param unit: The unit type, default=None
    :type unit: str, optional
    :param return_unit: If True, the function returns a tuple with the numeric value and the unit.
    :type return_unit: bool, optional
    :return: The converted value, optionally with the unit if `return_unit` is True.
    :rtype: float(tuple)
    """
    # If value is a string, split it to separate the value from unit
    if isinstance(value, str):
        try:
            numeric_val, unit_full = value.strip().split()
            numeric_val = float(numeric_val)
            final_value = numeric_val * _si_multiplier_unit(unit_full[0])
        except ValueError:
            print(f"Invalid value format: {value}. Should be '"
                             f"number unit'(Ex:'1.2 kHz')")

    # If value is a numer:
    elif isinstance(value, (int, float)):
        if unit is None:
            raise ValueError("Unit is not provided. Ex: 1.2, 'kHz'")
        final_value = value * _si_multiplier_unit(unit[0])
    else:
        raise TypeError("Provide the parameters!")
    if return_unit:
        return final_value, unit[1:]
    return final_value


def _get_prefix_val() -> dict:
    """
    Return a dictionary of SI prefixes and the corresponding multiplier values.

    This function is used internally to provide the SI prefix values.

    :return: A dictionary mapping SI prefixes to their values.
    :rtype: dict
    """

    prefix = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "μ": 1e-6,
        "m": 1e-3,
        "c": 1e-2,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
    }
    return prefix


def _si_multiplier_unit(unitstr: str) -> float:
    """
    Return a multiplier for a given unit prefix.

    Given a single character string (SI prefix) it
    returns the corresponding multiplier value.

    :param unitstr: A single character string = SI prefix.
    :type unitstr: str
    :return: The multiplier corresponding to the given SI prefix.
    :rtype: float
    """

    if unitstr in _get_prefix_val():
        return _get_prefix_val()[unitstr]

    raise ValueError(f"Unit prefix is not known: {unitstr}, see available:\n{_get_prefix_val()}")

@dataclass
class Filter:
    cutoff_frequency: float
    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"]
    filter_method: Literal["butter", "bessel"] = field(default="butter")
    order: int = field(default=2)
    bidirectional: bool = True
    sampling_frequency: float = None

    def __post_init__(self):
        if self.order < 1:
            self.order = 1

        if self.sampling_frequency:
            self._calculate_sos()

    def _calculate_sos(self):
        nyquist = 0.5 * self.sampling_frequency
        normalized_cutoff = self.cutoff_frequency / nyquist

        if self.filter_method == "butter":
            self.sos = signal.butter(self.order, normalized_cutoff,
                                     btype=self.filter_type,
                                     output='sos')
        elif self.filter_method == "bessel":
            self.sos = signal.bessel(self.order, normalized_cutoff,
                                     btype=self.filter_type,
                                     output='sos', norm='mag')
        else:
            raise ValueError(f"Unsupported filter method: {self.filter_method}")

    def __call__(self, current, sampling_frequency=None):
        if self.sampling_frequency is None and sampling_frequency is None:
            raise ValueError("Sampling frequency must be provided.")

        if sampling_frequency:
            self.sampling_frequency = sampling_frequency

        if not hasattr(self, 'sos'):
            self._calculate_sos()

        if self.bidirectional:
            current[:] = signal.sosfiltfilt(self.sos, current, axis=0)
        else:
            current[:] = signal.sosfilt(self.sos, current, axis=0)

        # TODO: add a copy of the file to avoid reopening the file


@dataclass
class Trimmer:
    samples_to_remove: int
    rank: str = "vstep"
    newrank: str = "vstepgap"

    def __call__(self, trace_file):
        """
        Trim segments within 'vstep' and create 'vstepgap' children.
        """
        for v in trace_file.traverse_to_rank(self.rank):
            if v.end - v.start > self.samples_to_remove:
                v.add_child(MetaSegment(
                    start=v.start + self.samples_to_remove,
                    end=v.end,
                    rank=self.newrank,
                    parent=v
                ))


def extract_features(seg, bottom_rank, extractions: list[str], add_ons: dict = {}, lambdas={}):
    """

    :param seg:
    :type seg: object
    :param bottom_rank:
    :type bottom_rank: str
    :param extractions:
    :param add_ons:
    :param lambdas:
    :return:
    """
    headers = extractions + list(add_ons.keys()) + list(lambdas.keys())

    df = pd.DataFrame(columns=headers)
    for bottom_seg in seg.traverse_to_rank(bottom_rank):
        row_dict = {}
        for feature in extractions:
            row_dict[feature] = bottom_seg.get_feature(feature)
        for feature, value in add_ons.items():
            row_dict[feature] = value
        for feature, lambda_func in lambdas.items():
            row_dict[feature] = lambda_func(bottom_seg)
        df.loc[len(df)] = row_dict
    return df
# extract_features(seg, bottom_rank='event',
#              extractions=['mean', 'frac', 'duration', 'baseline', 'current', 'wrap', 'start'],
#              add_ons={"sample_type": "MBP_D10"},
#              lambdas={"Voltage": lambda seg: int(1000 * seg.get_feature("voltage")),
#                       "start_time": lambda seg: seg.start / seg.get_feature("eff_sampling_freq")})



