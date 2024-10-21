#!/usr/bin/env python
"""
Module io.py manages IO process
"""
import xml.etree.ElementTree as ET
import os
import glob
import pyabf
import numpy as np
from ioniq.utils import si_eval
from ioniq.utils import split_voltage_steps


class AbstractFileReader(object):
    """
    An abstract class for reading various data files (.abf, .edh, .mat, .opt, .xml)
    This class defines the structure for file readers
    """
    # File extension. Replace with the appropriate file extension
    # in subclass, such as ".abf", ".edh", ".mat", etc.
    ext = "___"

    # Replace with a list of accepted keyword arguments passed
    # to the _read() function in every subclass.
    accepted_keywords = []

    # Values to scale voltage and current to SI units.
    # Change these in subclasses to match the idata scale
    current_multiplier: float = 1.0
    voltage_multiplier: float = 1.0

    def __init__(self):
        self.filename = "UNDEFINED"

    def read(self, filename: str, **kwargs):
        """
        Read a datafile or series of files. Files are identified according to their extension.
        Data formats that come with a header file must be referred to by the header file.
        If inheriting from the AbstractFileReader class, do not override this function;
        instead, create a custom _read method.

        :param filename: file name or list of file names, typically
        given as a string or PathLike object.
        :type filename: str or os.PathLike or list[str] or list[os.PathLike]
        :param **kwargs: keyword arguments passed directly to the file reader class
        that matches the data format.
        :return: [metadata, current, etc.. ]. If the input "filename" is a list,
        this function returns a generator
        object that yields the output of _read() for every file in the input list.
        :rtype: tuple[dict,np.ndarray [,np.ndarray or tuple[slice,np.float32]]]
        """
        for key in kwargs.keys():
            if key not in self.accepted_keywords:
                raise TypeError(f"{self.__class__}.read() got an unexpected argument: {key}")
        if type(filename) is list:
            self.kwargs = kwargs
            for fname in filename:
                assert os.path.splitext(fname)[-1].lower() == self.ext.lower()
            return (self._read(fname, **kwargs) for fname in filename)

        else:
            assert(os.path.splitext(filename)[-1].lower() == self.ext.lower())
            self.kwargs = kwargs
            return self._read(filename, **kwargs)

    def _read(self, filename, **kwargs):
        pass  # rewrite this function in inherited classes to process the data

    def __repr__(self):
        print(self.filename)


class EDHReader(AbstractFileReader):
    """
    A class for reading and processing .edh files, including parsing
    metadata and extracting current and voltage data

    """

    ext = ".edh"
    accepted_keywords = ["voltage_compress", "n_remove", "downsample", "prefilter"]

    current_multiplier = 1e-9  # current is stored in nA in the datafile
    voltage_multiplier = 1e-3  # voltage is stored in mV in the datafile

    def __init__(self):
        super().__init__()

    def _read(self, filename, **kwargs):
        filename = os.path.abspath(filename)
        direc = os.path.dirname(filename)
        metadata = {}

        with open(filename, 'r') as headerfile:
            for line in headerfile:
                lsplit = line.split(":")
                match lsplit:
                    case ["EDH Version" | "Channels" | "Oversampling x4" | "Active channels", *val]:
                        metadata[lsplit[0]] = "".join(val).strip()
                    case ["Sampling frequency (SR)" | "Range", *val]:
                        metadata[lsplit[0]] = si_eval("".join(val).strip())
                    case ["Final Bandwidth", *val]:
                        metadata["Final Bandwidth"] = metadata["Sampling frequency (SR)"] / \
                                                      int(val[0].strip().split()[0].split("/")[1])
                    case ["Acquisition start time", *val]:
                        metadata["Acquisition start time"] = " ".join(line.split(" ")[-2:])
                        # print(metadata)
                    case _:
                        pass
        # print({metadata})
        # if multichannel:
        #     active_channels= list(map(int,metadata["Active channels"].split()))
        #     active_channels=[str(x-1) for x in active_channels]
        #     core_fname=os.path.splitext(os.path.split(filename)[-1])[0]
        #     for channel_name in active_channels:
        #         file_list_abf=glob.glob(f"{core_fname}_CH00{channel_name}_*.abf",root_dir=direc)

        file_list_abf = glob.glob("*.abf", root_dir=direc)

        if len(file_list_abf) > 0:
            abf_buffers = tuple(map(pyabf.ABF, [os.path.join(direc, file)
                                                for file in file_list_abf]))
            # list(map())
            current = np.concatenate([buffer.data[0] for buffer in abf_buffers],
                                     axis=0, dtype=np.float32)
            voltage = np.concatenate([buffer.data[-1] for buffer in abf_buffers],
                                     axis=0, dtype=np.float32)
            metadata["DataFiles"] = file_list_abf
            metadata["StorageFormat"] = ".abf"
        else:
            file_list_dat = glob.glob("*.dat", root_dir=direc)
            if len(file_list_dat) == 0:
                raise FileExistsError("No associated data files (*.abf or *.dat) found.")
            data = np.concatenate([np.fromfile(os.path.join(direc, file), dtype="float32")
                                   for file in file_list_dat])
            data = data.reshape((int(metadata["Active channels"])+1, -1), order="F")
            current = data[0]
            voltage = data[-1]
            metadata["DataFiles"] = file_list_dat
            metadata["StorageFormat"] = ".dat"
        assert current.shape == voltage.shape
        metadata["HeaderFile"] = filename

        # Scale the current and voltage arrays to SI units
        if kwargs.get("prefilter", None):
            prefilter = kwargs.get("prefilter")
            assert callable(prefilter)
            prefilter(current)

        current *= self.current_multiplier
        voltage *= self.voltage_multiplier
        if kwargs.get("downsample", None):
            downsample_factor = kwargs.get("downsample")
            assert type(downsample_factor) is int, \
                f"non-integer downsampling factor not supported:" \
                f"{type(downsample_factor)}, {downsample_factor}"
            if downsample_factor > 1:
                _current = current[::downsample_factor].copy()
                _voltage = voltage[::downsample_factor].copy()
                del current, voltage
                current, voltage = _current, _voltage
                metadata["downsample"] = downsample_factor
                metadata["eff_sampling_freq"] = \
                    metadata["Sampling frequency (SR)"] / downsample_factor

        if kwargs.get("voltage_compress", False):
            n_remove = kwargs.get("n_remove", 0)
            voltage_splits = split_voltage_steps(voltage, as_tuples=True, n_remove=n_remove)
            voltage_points = [(sl, voltage[sl[0]]) for sl in voltage_splits]
            del voltage

            return metadata, current, voltage_points

        return metadata, current, voltage


class OPTReader(AbstractFileReader):
    """
    Class to handle OPT  and XML files from Axopatch
    to get a shift timing between voltage and current
    """
    ext = ".opt"

    def __init__(self):
        """
        Parent AbstractFileReader class initializations
        """

    def _read_opt(self, filename):
        """

        :param filename:
        :return:
        """
        # is opt file binary?
        file = np.fromfile(filename, dtype="float32")

    def _read_xml(self, filename):
        """

        :param filename:
        :return:
        """
        tree = ET.parse(filename)
        root = tree.getroot()

        timing_shift_raw = root.find("timestamp")
        if timing_shift_raw is not None:
            timing_shift = float(timing_shift_raw.text)
        else:
            timing_shift = 0
        return timing_shift


if __name__ == "__main__":
    # print(EDHReader.ext)
    e = EDHReader()
    meta, current, voltage = e.read("../../tests/data/8e7_80n01M1_5pctSorbitol_IV/"
                                    "8e7_80n01M1_5pctSorbitol_IV.edh", voltage_compress=True)
    print(len(voltage))
    # import matplotlib.pyplot as plt
    # plt.plot(current[::100])
    # plt.waitforbuttonpress()
    # e.read("C:/Users/alito/EDR/Q402m1_SBead/Q402m1_SBead.edh")

    ###################################
    #  Explore XML files
    ####################################
    # tree = ET.parse("../../test_data/TOKW1_DPhPC_Chol_Hexane/B090524SR_100kHz__000.xml")
    # root = tree.getroot()
    #
    # timing_data = []
    # sweep_data = []
    #
    # # timestamp and alignment inf from XML
    # for timestamp in root.findall('timestamp'):
    #     wall_clock = float(timestamp.get('wall_clock'))
    #     msec = int(timestamp.get('msec'))
    #
    #     # sweep information
    #     sweep = timestamp.find('sweep')
    #     if sweep is not None:
    #         sweep_number = int(sweep.get('N'))
    #         sweep_data.append({'sweep': sweep_number, 'time': wall_clock + msec / 1000})
    #
    #     # HWtiming_cap_step inside timestamp
    #     hw_timing = timestamp.find('HWtiming_cap_step')
    #     if hw_timing is not None:
    #         # time_alignment_marks
    #         alignment_segments = hw_timing.find('time_alignment_marks')
    #         if alignment_segments is not None:
    #             segments = alignment_segments.findall('time_alignment_segment')
    #
    #             # Extract information from each time_alignment_segment
    #             for segment in segments:
    #                 num_samples = int(segment.get('number_samples'))
    #                 voltage_mV = float(segment.get('voltage_mV'))
    #                 time_ms = float(segment.get('time_ms'))
    #
    #                 # Store the segment information in the list
    #                 timing_data.append({
    #                     'samples': num_samples,
    #                     'voltage': voltage_mV,
    #                     'time_ms': time_ms
    #                 })
    #
    # # Output the results
    # print(f"Number of alignment segments: {len(timing_data)}")
    # for segment in timing_data:
    #     print(segment)
    #
    # opt_file = np.fromfile("../../test_data/TOKW1_DPhPC_Chol_Hexane/"
    #                        "B090524SR_100kHz__000.opt", dtype="float32")
    # print(opt_file[1:10])
