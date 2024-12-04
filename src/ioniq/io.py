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
from ioniq.setup_log import json_logger
from ioniq.datatypes import SessionFileManager
import uuid
from scipy.signal import find_peaks


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
        self.uuid = str(uuid.uuid4())
        sfm = SessionFileManager()
        sfm.register_affector(self)
    def __repr__(self):
        return f"<AbstractFileReader UUID = {self.uuid}"

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

    @json_logger.log
    def __init__(self, filename, voltage_compress=False, n_remove=0, downsample=1, prefilter=None):
        super().__init__()
        self.filename = filename
        self.voltage_compress = voltage_compress
        self.n_remove = n_remove
        self.downsample = downsample
        self.prefilter = prefilter
        self.metadata, self.current, self.voltage = self._read()

    def __iter__(self):
        return iter((self.metadata, self.current, self.voltage))

    def _read(self):
        filename = os.path.abspath(self.filename)

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

        if self.prefilter:
            assert callable(self.prefilter)
            self.prefilter(current)

        current *= self.current_multiplier
        voltage *= self.voltage_multiplier

        if self.downsample > 1:
            current_data = current[::self.downsample]
            voltage_data = voltage[::self.downsample]
            metadata["downsample"] = self.downsample
            metadata["eff_sampling_freq"] = metadata["Sampling frequency (SR)"] / self.downsample

        if self.voltage_compress:
            voltage_splits = split_voltage_steps(voltage, as_tuples=True, n_remove=self.n_remove)
            voltage_points = [(sl, voltage[sl[0]]) for sl in voltage_splits]
            del voltage
            return metadata, current, voltage_points

        return metadata, current, voltage


class XMLReader(AbstractFileReader):
    """
    A class for reading and processing .opt files, including parsing
    metadata and extracting current and voltage data.
    """

    ext = ".opt"
    accepted_keywords = ["voltage_compress", "n_remove", "downsample", "prefilter"]
    current_multiplier = 1e9  # Convert current to nA

    def __init__(self, xml_filename: str, voltage_compress=False, n_remove=0, downsample=1, prefilter=None):
        super().__init__()
        self.xml_filename = xml_filename
        self.voltage_compress = voltage_compress
        self.n_remove = n_remove
        self.downsample = downsample
        self.prefilter = prefilter
        # Find opt file
        base_name = os.path.splitext(os.path.basename(xml_filename))[0]
        direc = os.path.dirname(xml_filename)
        pattern = os.path.join(direc, f"{base_name}.opt")
        self.opt_filename = glob.glob(pattern)[0]
        self.metadata, self.current, self.voltage = self._read()

    def __iter__(self):
        return iter((self.metadata, self.current, self.voltage))

    def _read(self):
        metadata = self._parse_xml_metadata()
        current_full = self._load_opt_data()
        voltage = self._align_voltage(metadata, current_full)
        current = current_full

        if self.prefilter:
            assert callable(self.prefilter)
            self.prefilter(current)

        current *= self.current_multiplier

        if self.downsample > 1:
            current = current[::self.downsample]
            voltage = voltage[::self.downsample]
            metadata["downsample"] = self.downsample
            metadata["eff_sampling_freq"] = metadata["Sampling frequency (SR)"] / self.downsample

        if self.voltage_compress:
            voltage_splits = split_voltage_steps(voltage, as_tuples=True, n_remove=self.n_remove)
            voltage_points = [(sl, voltage[sl[0]]) for sl in voltage_splits]
            del voltage
            return metadata, current, voltage_points

        return metadata, current, voltage

    def _parse_xml_metadata(self):
        """
        Parses the XML file and extracts metadata for the entire experiment.
        """
        try:
            tree = ET.parse(self.xml_filename)
            root = tree.getroot()
        except (FileNotFoundError, ET.ParseError) as e:
            raise IOError(f"Error reading XML file {self.xml_filename}: {e}")

        metadata = {
            "HeaderFile": os.path.abspath(self.xml_filename),
            "downsample": self.downsample,
        }

        metadata.update(self._extract_sampling_info(root))
        metadata.update(self._extract_acquisition_time(root))
        metadata.update(self._extract_file_info())
        metadata.update(self._calculate_bandwidth(metadata))
        return metadata

    def _extract_sampling_info(self, root):
        """
        Extracts sampling frequency, total samples, and total time from the XML.
        """
        hw_timing = root.find(".//HWtiming_cap_step")
        hw_timing_1 = hw_timing.find("cap_step_waveform")

        if hw_timing is None or hw_timing_1 is None:
            raise ValueError("HWtiming_cap_step or cap_step_waveform not found in XML.")

        sample_rate = float(hw_timing.get("sample_rate_Hz"))
        total_samples = int(hw_timing_1.get("number_samples"))
        total_time = total_samples / sample_rate

        return {
            "Sampling frequency (SR)": sample_rate,
            "total_samples": total_samples,
            "total_time_s": total_time}

    def _extract_acquisition_time(self, root):
        """
        Get acquisition start time from the XML.
        """
        start_time = root.find(".//timestamp")
        if start_time is not None and "wall_clock" in start_time.attrib:
            return {"Acquisition start time": start_time.attrib["wall_clock"]}
        return {}

    def _extract_file_info(self):
        """
        Get filename metadata.
        """
        return {
            "DataFiles": [os.path.basename(self.xml_filename)],
            "StorageFormat": os.path.splitext(self.xml_filename)[-1]}

    def _calculate_bandwidth(self, metadata):
        """
        Final bandwidth based on sampling frequency.
        """
        sampling_frequency = metadata["Sampling frequency (SR)"]
        return {"Final Bandwidth": sampling_frequency / 2 if sampling_frequency < 200000 else 100000}

    def _load_opt_data(self):
        """
        Reads the current data from the .opt file.
        """
        try:
            dtype = np.dtype(">d")
            current = np.fromfile(self.opt_filename, dtype)
            return current
        except Exception as e:
            raise IOError(f"Error reading OPT file {self.opt_filename}: {e}")

    def _align_voltage(self, metadata, current_full):
        """
        Aligns the voltage to the current signal using XML metadata, starting
        from the first detected peak in the current data.
        """
        if not hasattr(self, "_xml_tree"):
            self._xml_tree = ET.parse(self.xml_filename).getroot()
        root = self._xml_tree
        total_samples = metadata["total_samples"]
        sample_rate = metadata["Sampling frequency (SR)"]

        # Calculate the initial start sample
        start_sample = self._get_start_sample(root, sample_rate)
        print(start_sample)

        # Initialize voltage array
        voltage_waveform = np.zeros(len(current_full), dtype=np.float32)

        # Step 2: Dynamically determine the search range for the first peak
        time_marks = root.find(".//HWtiming_cap_step/time_alignment_marks")
        if time_marks is None:
            raise ValueError("Time alignment marks not found in the XML.")

        # TODO: Should we change to a certain value the end limit of the find_peak search?
        search_end = start_sample + 200000

        # Detect the first peak in the segment
        peaks, properties = self.find_peaks_in_segment(current_full, start_sample, search_end)

        if not peaks.size:
            raise ValueError("No peaks detected in the current data for alignment.")

        first_peak_index = peaks[0] + start_sample
        # print("first_peak_index", first_peak_index)

        # Process time alignment marks, adjusting for zero-voltage samples
        current_index = self._process_time_marks(root, voltage_waveform, first_peak_index)

        # print(f"idx after processing time marks: {current_index}")

        # Process cap step waveform
        self._process_cap_step_waveform(root, voltage_waveform, current_index)

        return voltage_waveform

    def _get_start_sample(self, root, sample_rate):
        """
        Retrieves the starting sample index based on timestamp information.
        """
        last_msec = None
        for elem in root.iter():
            if elem.tag == "timestamp" and "msec" in elem.attrib:
                last_msec = float(elem.attrib["msec"])
            elif elem.tag == "HWtiming_cap_step":
                break

        if last_msec is None:
            raise ValueError("No <timestamp> with 'msec' attribute found.")
        msec = int(last_msec * sample_rate / 1000)

        return msec

    def _process_time_marks(self, root, voltage_waveform, first_peak_index):
        """
        Processes time alignment marks to get the voltage waveform.
        The voltage alignment starts at current_index[first_peak_index - zero_voltage_samples].
        """

        time_marks = root.find(".//HWtiming_cap_step/time_alignment_marks")
        zero_voltage_samples = 0
        if time_marks is not None:
            # get the samples before the first peak (zero voltage samples)
            for segment in time_marks.findall("time_alignment_segment"):
                number_samples = int(segment.get("number_samples"))
                voltage_mV = float(segment.get("voltage_mV"))

                if voltage_mV == 0.0:
                    zero_voltage_samples += number_samples
                else:
                    break

            # shift the starting idx
            start_index = max(first_peak_index - zero_voltage_samples, 0)

            # Now process all time alignment segments
            for segment in time_marks.findall("time_alignment_segment"):
                number_samples = int(segment.get("number_samples"))
                voltage_mv = float(segment.get("voltage_mV"))
                # print("voltage_mv", voltage_mv, "start_index ", start_index, "start_index + number of samples", start_index + number_samples)
                # Assign voltage values starting from the adjusted index
                voltage_waveform[start_index:start_index + number_samples] = voltage_mv
                # print(f"num_samples {number_samples} with voltage {voltage_mv}. start from: {start_index}")

                start_index += number_samples

        return start_index

    def _process_cap_step_waveform(self, root, voltage_waveform, current_index):
        """
        Processes the cap step waveform to add triangle wave segments.
        """
        cap_waveform = root.find(".//HWtiming_cap_step/cap_step_waveform")
        if cap_waveform is None:
            return

        leading_samples = int(cap_waveform.get("leading_number_samples", 0))
        trailing_samples = int(cap_waveform.get("trailing_number_samples", 0))

        first_triangle = cap_waveform.find(".//triangle_wave")
        if first_triangle is not None:
            leading_offset = float(first_triangle.get("offset_mV"))
            voltage_waveform[current_index:current_index + leading_samples + trailing_samples] = leading_offset

        for triangle in cap_waveform.findall(".//triangle_wave"):
            offset_mV = float(triangle.get("offset_mV"))
            total_N_sample = int(triangle.get("total_N_sample"))
            end_index = current_index + total_N_sample + trailing_samples + leading_samples
            voltage_waveform[current_index:end_index] = offset_mV
            current_index = end_index

    def find_peaks_in_segment(self, current_data, start_index, end_index):
        """
        Apply "find_peaks" on a segment of the current data and return
        the indices of the found peak
        """
        # slice original data
        segment = current_data[start_index:end_index]

        peaks, properties = find_peaks(segment, height=1e-9)
        # print(peaks)

        return peaks, properties


if __name__ == "__main__":
    # print(EDHReader.ext)
    # e = EDHReader()
    # meta, current, voltage = e.read("../../tests/data/8e7_80n01M1_5pctSorbitol_IV/"
    #                                 "8e7_80n01M1_5pctSorbitol_IV.edh", voltage_compress=True)

    # import matplotlib.pyplot as plt
    # plt.plot(current[::100])
    # plt.waitforbuttonpress()
    # e.read("C:/Users/alito/EDR/Q402m1_SBead/Q402m1_SBead.edh")
    #xml_file = "/Users/dinaraboyko/grad_school/cloned_repo/data/TOKW/B090624SR_100kHz__000.xml"
    xml_file = "/Users/dinaraboyko/grad_school/cloned_repo/data/TOKW/B082224SR_250kHz__003.xml"
    reader = XMLReader(xml_file, voltage_compress=True, downsample=1)
    metadata, current, voltage = reader
    #
    print("Metadata:", metadata)
    print("Curren:", len(current))
    print("Voltage:", voltage)






