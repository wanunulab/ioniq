#!/usr/bin/env python
"""
Some of the parsers and the parser base class were adapted from the
PyPore Package by Jacob Schreiber and Kevin Karplus (https://github.com/jmschrei/PyPore)
"""


# import sys
# from itertools import tee, chain
# import re

try:
    import cupy
    import cupyx.scipy.signal as signal
    if not cupy.cuda.is_available():
        raise ImportError

    np = cupy
except ImportError:

    import numpy as np
    from scipy import signal

from typing import TypeVar
from itertools import tee
# import time
import json
from scipy import signal
from scipy.fft import fft
import numpy as np
import pyximport
from ioniq.core import Segment, MetaSegment
import abc
from ioniq.cparsers import FastStatSplit
pyximport.install(setup_args={'include_dirs': np.get_include()})
AnyParser = TypeVar("AnyParser", bound="Parser")


#########################################
# EVENT PARSERS
#########################################


class Parser(abc.ABC):
    """
    Class for parsing segments of data, used for processing
    segments of current data
    """
    # override this with the required attributes to get
    # from the parent segment that the parser needs
    required_parent_attributes: list[str] = ['current']
    @abc.abstractmethod
    def __init__(self):
        """
        Initialize a Parser instance
        """
        pass

    @classmethod
    def get_name(cls):
        """
        Get the name of the parser class
        :return: The name of the class.
        :rtype: str
        """
        return cls.__name__

    @classmethod
    def get_init_params(cls):
        """
        Get the initialization parameters of the parser class.

        :return: A dictionary of initialization parameters, default None.
        :rtype: dict or None
        """
        return getattr(cls, "_params", None)

    @classmethod
    def get_process_inputs(cls):
        return getattr(cls, "_parse_inputs", None)

    @classmethod
    def get_process_outputs(cls):
        """
        Get the outputs generated by processing

        :return: A list of outputs generated by parse method
        :rtype: list or None
        """
        return getattr(cls, "_parse_outputs", None)

    # @classmethod
    def get_required_parent_attributes(self):
        """
        Get the list of required attributes from the parent segment

        :return: A list of required parent attributes
        :rtype: list[str]
        """
        return getattr(self, "required_parent_attributes", [])

    def __repr__(self):
        """
        Returns a representation of the parser in the form of all arguments.

        :return: A JSON representation of the parser's attributes.
        :rtype: str

        """
        return self.to_json()

    def to_dict(self):
        """
        Convert the parser's attributes to a dictionary.

        :return: A dictionary representation of the parser's attributes.
        :rtype: dict
        """

        dict_attr = {key: val for key, val in list(self.__dict__.items())
                     if key != 'param_dict' if type(val) in (int, float)
                     or ('Qt' not in repr(val)) and 'lambda' not in repr(val)}
        dict_attr['name'] = self.__class__.__name__
        return dict_attr

    def to_json(self, filename=False):
        """
        Convert the parser's attributes to a JSON string.

        :param filename: If filename, it writes the JSON string to the specified file.
        :type filename: str, default bool
        :return: A dictionary representation of the parser's attributes.
        :rtype: str
        """

        _json = json.dumps(self.to_dict(), indent=4, separators=(',', ' : '))
        if filename:
            with open(filename, 'r') as out:
                out.write(_json)
        return _json

    def parse(self, **kwargs):
        """
        Takes in a current segment, and returns a list of segment objects

        :param kwargs: Keyword arguments containing the data, such as current
        :type kwargs: dict
        :return: A list of tuples representing the start and end indices of the segment
                 and an empty dictionary.
        :rtype: list[tuple]
        """
        current = kwargs['current']
        return [(0, current.shape[0], {})]
        # return [ Segment( current=current, start=0, duration=current.shape[0]/100000 ) ]

    def set_params(self, param_dict):
        """
        Updates each paramater presented in the GUI to the value input in the lineEdit
        corresponding to that value.
        :param param_dict: A dictionary
        :type param_dict: dict
        """
        try:
            # for key, lineEdit in list(self.param_dict.items()):
            #     val = lineEdit.text()
            #     if '.' in val:
            #         setattr( self, key, float( val ) )
            #         continue
            #     for i, letter in enumerate(val):
            #         if not letter.isdigit():
            #             setattr( self, key, str( val ) )
            #             continue
            #         if i == len(val):
            #             setattr( self, key, int( val ) )
            self.param_dict = param_dict
            for key, value in self.param_dict.items():
                setattr(self, key, value)
        except:  # Add Error?
            pass

    @classmethod
    def from_json(self, _json):
        """
        Critical: Not implemented yet, figure out how to make this work in pythion"""
        if _json.endswith(".json"):
            with open(_json, 'r') as infile:
                _json = ''.join(line for line in infile)

        d = json.loads(_json)
        name = d['name']
        del d['name']

        return None  # getattr( PyPore.parsers, name )( **d )


class SpikeParser(Parser):
    required_parent_attributes = ["current", "sampling_freq", "mean", "start", "std"]
    _fractionable_attrs = ['height', 'threshold', 'prominence']
    _time_attrs = ['distance', 'prominence', 'width', 'wlen', 'plateau_size']
    _other_attrs = ['rel_height']

    def __init__(self, height=None, threshold=None, distance=None,
                 prominence=None, prominence_snr=None, width=None,
                 wlen=None, rel_height: float = 0.5, plateu_size=None,
                 fractional: bool = True) -> None:
        # Add super()__init__ to isinitate class PArser?
        #super().__init__()
        self._height = height
        self._threshold = threshold
        self._distance = distance
        self._prominence = prominence
        self._prominence_snr = prominence_snr
        self._width = width
        self._wlen = wlen
        self._rel_height = rel_height
        self._plateau_size = plateu_size
        self._fractional = fractional

    def _calculate_absolute_from_fractionals(self, mean):
        pass

    def parse(self, current, sampling_freq, mean, std, start):
        from scipy import signal
        abs_start = start
        for key in self._time_attrs:
            _key = '_' + key
            attr = getattr(self, _key, None)
            # if attr != None:
            if attr is not None:
                if type(attr) in [tuple, list]:
                    # print(sampling_freq)
                    setattr(self, key,
                            (sampling_freq * attr[0] if attr[0] is not None else None,
                             sampling_freq * attr[1] if attr[0] is not None else None))
                else:
                    setattr(self, key, sampling_freq * attr)
            else:
                setattr(self, key, attr)

        for key in self._fractionable_attrs:
            _key = '_' + key
            attr = getattr(self, _key, None)
            if self._fractional:
                # if attr !=None:
                if attr is not None:
                    if type(attr) in [tuple, list]:
                        setattr(self, key,
                                (mean * attr[0] if attr[0] is not None else None,
                                 mean * attr[1] if attr[0] is not None else None))
                    else:
                        setattr(self, key, mean * attr)
                else:
                    setattr(self, key, attr)
            else:
                setattr(self, key, attr)

        if self._prominence_snr is not None and self._fractional:
            if type(self._prominence_snr) in [tuple, list]:
                try:
                    iticks = np.linspace(0, len(current), 11).astype(int)
                    std_rolling = np.array([np.std(current[ii:jj])
                                            for ii, jj in zip(iticks[:-1], iticks[1:])])
                    mean_rolling = np.array([np.mean(current[ii:jj])
                                             for ii, jj in zip(iticks[:-1], iticks[1:])])
                    std_1stq = np.quantile(std_rolling, q=0.25)
                    mean_1stq = np.quantile(mean_rolling, q=0.25)
                    level = std_1stq / np.sqrt(np.abs(mean_1stq))
                    self.prominence = (self._prominence_snr[0] * level
                                       if self._prominence_snr[0] is not None else None,
                                       self._prominence_snr[1] * np.abs(mean_1stq)
                                       if self._prominence_snr[1] is not None
                                       else self.prominence[1]
                                       if type(self.prominence) in [list, tuple] else None)
                except Exception as error:
                    print("PARSING ERROR:\n", error)
                    print("using whole mean/std instead of median of rolling...")
                    level = std / np.sqrt(np.abs(mean))
                    self.prominence = (self._prominence_snr[0] * level
                                       if self._prominence_snr[0] is not None else None,
                                       self._prominence_snr[1] * np.abs(mean)
                                       if self._prominence_snr[1] is not None else
                                       self.prominence[1]
                                       if type(self.prominence) in [list, tuple] else None)

        for key in self._other_attrs:
            setattr(self, key, getattr(self, "_" + key))

        all_keys = self._fractionable_attrs + self._time_attrs + self._other_attrs
        find_peaks_args = dict([(key, getattr(self, key)) for key in all_keys])
        peaks, props = signal.find_peaks(current, ** find_peaks_args)
        dt = np.array(0)
        dt = np.append(dt, np.diff(peaks) / sampling_freq)
        for i, peak in enumerate(peaks):
            event_start = round(props["left_ips"][i])
            event_end = round(props["right_ips"][i])
            unique_features = {
                "idx_rel": peak,
                "idx_abs": abs_start + peak,
                "peak_val": current[peak],
                "deli": props["prominences"][i],
                "dwell": props["widths"][i] * 1e6 / sampling_freq,
                "dt": dt[i],
                "frac": props["prominences"][i] / mean}
            yield event_start, event_end, unique_features
        return


class MemoryParse(object):
    """
    A parser based on being fed previous split points, and splitting a raw file based
    those splits. Used predominately when loading previous split points from the
    database cache, to reconstruct a parsed file from "memory.
    """
    def __init__(self, starts, ends):
        self.starts = starts
        self.ends = ends

    def parse(self, current):
        return [Segment(current=np.array(current[int(s): int(e)], copy=True),
                        start=s, duration=e - s) for s, e in zip(self.starts, self.ends)]


class lambda_event_parser(Parser):  # Rename class to Upper case!
    """
    A simple rule-based parser which defines events as a sequential series
    of points which are below a certain threshold, then filtered based on other
    critereon such as total time or minimum current. Rules can be passed in at
    initiation, or set later, but must be a lambda function takes in a PreEvent
    object and performs some boolean operation.
    """

    def __init__(self, threshold=90, rules=None):
        super().__init__()  # Add super()__init__ to isinitate class PArser?
        self.threshold = threshold
        self.rules = rules or [lambda event: event.duration > 100000,
                               lambda event: event.min > -0.5,
                               lambda event: event.max < self.threshold]

    def _lambda_select(self, events):
        """
        From all of the events, filter based on whatever set of rules has been initiated with.
        """
        return [event for event in events if np.all([rule(event) for rule in self.rules])]

    def parse(self, current):
        """
        Perform a large capture of events by creating a boolean mask for when
        the current is below a threshold, then detecting the edges in those masks,
        and using the edges to partitition the sample. The events are
        then filtered before being returned.
        """
        # Find where the current is below a threshold, replace with 1's
        mask = np.where(current < self.threshold, 1, 0)
        # Find the edges, marking them with a 1, by derivative
        mask = np.abs(np.diff(mask))
        tics = np.concatenate(([0], np.where(mask == 1)[0] + 1, [current.shape[0]]))
        del mask
        events = [Segment(current=np.array(current), copy=True, start=tics[i],
                          duration=current.shape[0])
                  for i, current in enumerate(np.split(current, tics[1:-1]))]
        return [event for event in self._lambda_select(events)]

    def set_params(self):
        """
        Read in the data from the GUI and use it to customize
        the rules or threshold of the parser.
        """
        self.rules = []
        self.threshold = float(self.threshInput.text())
        self.rules.append(lambda event: event.max < self.threshold)
        if self.minCurrentInput.text() != '':
            self.rules.append(lambda event: event.min > float(self.minCurrentInput.text()))
        if self.timeInput.text() != '':
            if str(self.timeDirectionInput.currentText()) == '<':
                self.rules.append(lambda event: event.duration < float(self.timeInput.text()))
            elif str(self.timeDirectionInput.currentText()) == '>':
                self.rules.append(lambda event: event.duration > float(self.timeInput.text()))
        if self.rules == []:
            self.rules = None


def pairwise(iterable):
    """
     s -> (s0,s1), (s1,s2), (s2, s3), ...
    :param iterable:
    :return:
    """
    cur_item, next_item = tee(iterable)
    next(next_item, None)
    return zip(cur_item, next_item)


class SpeedyStatSplit(Parser):
    """
    See cparsers.pyx FastStatSplit for full documentation. This is just a
    wrapper for the cyton implementation to add a GUI.
    """
    uiTemplate = [("sampling_freq", "spin",
                   {'value': 200000, 'step': 1, 'dec': True, 'bounds': [0, None],
                    'suffix': 'Hz', 'siPrefix': True}),
                  ("min_width", "spin", {"value": 100, 'step': 1, 'dec': True,
                                         'min': 2, "int": True}),
                  ("max_width", "spin", {"value": 1000000, 'step': 1, 'dec': True, "int": True}),
                  ("window_width", "spin", {"value": 10000, 'step': 1, 'dec': True, "int": True}),
                  #  ("min_gain_per_sample", "(float,None,"hidden"),
                  #  "false_positive_rate":(float,None,"hidden")
                  #  "prior_segments_per_second":(float,None,"hidden"),
                  ("cutoff_freq", "spin", {"hidden": True})]
    _parse_return_type = (list, Segment)
    _parse_meta_return_type = (list, MetaSegment)

    _parse_inputs = [("Current", np.ndarray)]
    _parse_outputs = [("Segments", list)]

    def __init__(self, sampling_freq, min_width=100, max_width=1000000, window_width=10000,
                 min_gain_per_sample=None, false_positive_rate=None,
                 prior_segments_per_second=None, cutoff_freq=None):
        """
        Initialize the SpeedyStatSplit instance

        :param sampling_freq: The sampling frequency of the data
        :type sampling_freq: int
        :param min_width: Minimum width of a segment, default=100
        :type min_width: int, optional
        :param max_width: Maximum width of a segment, default=1000000.
        :type max_width: int, optional
        :param window_width: Width of the window, defaults to 10000.
        :type window_width: int, optional
        :param min_gain_per_sample: Minimum gain per sample, default=None
        :type min_gain_per_sample: float or None, optional
        :param false_positive_rate: Expected false positive rate , default=None
        :type false_positive_rate: float or None, optional
        :param prior_segments_per_second: Prior number of segments per second, default=None.
        :type prior_segments_per_second: float or None, optional
        :param cutoff_freq: Cutoff frequency for filtering, default=None
        :type cutoff_freq: float or None, optional
        """

        self.min_width = min_width
        self.max_width = max_width
        self.min_gain_per_sample = min_gain_per_sample
        self.window_width = window_width
        self.prior_segments_per_second = prior_segments_per_second
        self.false_positive_rate = false_positive_rate
        self.sampling_freq = sampling_freq
        self.cutoff_freq = cutoff_freq

    def parse(self, current):
        """
        Parse the current data to detect segments using the FastStatSplit

        :param current: The current data to be parsed
        :type current: numpy.ndarray
        :return: A list of detected segments
        :rtype:
        """
        parser = FastStatSplit(self.min_width, self.max_width,
                               self.window_width, self.min_gain_per_sample,
                               self.false_positive_rate, self.prior_segments_per_second,
                               self.sampling_freq, self.cutoff_freq)
        results=parser.parse(current)

        #print([(seg.start,seg.end,{}) for seg in results])
        return [(seg.start,seg.end,{}) for seg in results]

    def parse_meta(self, current):
        """
        Parse the current data and extract meta information

        :param current: The current data to be parsed.
        :type current: numpy.ndarray
        """
        parser = FastStatSplit(self.min_width, self.max_width, self.window_width,
                               self.min_gain_per_sample, self.false_positive_rate,
                               self.prior_segments_per_second, self.sampling_freq,
                               self.cutoff_freq)

        return parser.parse_meta(current)

    def best_single_split(self, current):
        """
        Determine the best single split point in the current data

        :param current: The current data to be analyzed.
        :type current: numpy.ndarray
        :return: The best split point
        :rtype: int or None
        """
        parser = FastStatSplit(self.min_width, self.max_width,
                               self.window_width, self.min_gain_per_sample,
                               self.false_positive_rate,
                               self.prior_segments_per_second, self.sampling_freq)
        return parser.best_single_split(current)

    def set_params(self):
        """
        Set the parameters
        """
        try:
            self.min_width = int(self.minWidth.text())
            self.max_width = int(self.maxWidth.text())
            self.window_width = int(self.windowWidth.text())
            self.min_gain_per_sample = float(self.minGain.text())
        except:
            pass

#########################################
# STATE PARSERS
#########################################


class snakebase_parser(Parser):  # Upper case for class name!
    """
    A simple parser based on dividing when the peak-to-peak
    amplitude of a wave exceeds a certain threshold.
    """

    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def parse(self, current):
        # Take the derivative of the current first
        diff = np.abs(np.diff(current))
        # Find the places where the derivative is low
        tics = np.concatenate(([0], np.where(diff < 1e-3)[0], [diff.shape[0]]))
        # For pieces between these tics, make each point the cumulative
        # sum of that piece and put it together piecewise
        cumsum = np.concatenate(([np.cumsum(diff[tics[i]: tics[i + 1]])
                                  for i in range(tics.shape[0] - 1)]))
        # Find the edges where the cumulative sum passes a threshold
        split_points = np.where(np.abs(np.diff(np.where(cumsum >
                                                        self.threshold, 1, 0))) == 1)[0] + 1
        # Return segments which do pass the threshold
        return [Segment(current=current[tics[i]: tics[i + 1]], start=tics[i])
                for i in range(1, tics.shape[0] - 1, 2)]

    def set_params(self):
        self.threshold = float(self.threshInput.text())


class FilterDerivativeSegmenter(Parser):
    """
    This parser will segment an event using a filter-derivative method. It will
    first apply a bessel filter at a certain cutoff to the current, then it will
    take the derivative of that, and segment when the derivative passes a
    threshold.
    """

    def __init__(self, low_threshold=1, high_threshold=2, cutoff_freq=1000., sampling_freq=1.e5):
        super().__init__()  # Add super()__init__ to isinitate class PArser?
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq

    def parse(self, current):
        """
        Apply the filter-derivative method to filter the ionic current.
        """

        # Filter the current using a first order Bessel filter twice, one in
        # both directions to preserve phase
        nyquist = self.sampling_freq / 2.
        numerator, denominator = signal.bessel(1, self.cutoff_freq /
                                               nyquist, btype='low', analog=0, output='ba')
        filtered_current = signal.filtfilt(numerator, denominator, np.array(current).copy())

        # Take the derivative
        deriv = np.abs(np.diff(filtered_current))

        # Find the edges of the blocks which fulfill pass the lower threshold
        blocks = np.where(deriv > self.low_threshold, 1, 0)
        block_edges = np.abs(np.diff(blocks))
        tics = np.where(block_edges == 1)[0] + 1

        # Split points are points in the each block which pass the high
        # threshold, with a maximum of one per block
        split_points = [0]

        # for start, end in it.izip( tics[:-1:2], tics[1::2] ): # For all pairs of edges for a block
        # izip is not in python3
        for start, end in zip(tics[:-1:2], tics[1::2]):  # For all pairs of edges for a block
            segment = deriv[start:end]  # Save all derivatives in that block to a segment
            # If the max derivative in that block is above a threshold
            if np.argmax(segment) > self.high_threshold:
                # Save the edges of the segment
                split_points = np.concatenate((split_points, [start, end]))

        # Now you have the edges of all transitions saved, and so the states are
        # the current between these transitions
        tics = np.concatenate((split_points, [current.shape[0]]))
        tics = list(map(int, tics))
        return [Segment(current=current[tics[i]:tics[i + 1]], start=tics[i])
                for i in range(0, len(tics) - 1, 2)]

    def set_params(self):
        """

        :return:
        """
        self.low_thresh = float(self.lowThreshInput.text())
        self.high_thresh = float(self.highThreshInput.text())


class NoiseFilterParser(Parser):
    """
    This unit is a parser subclass. when invoked on a segment (whole file or subsegment),
    it identifies the areas where bad things happened, and areas where the data is clean.
    Examples of bad things include high noise (60Hz noise from opening the cage), membrane
    rupture, and membrane formation attempts. it produces children from the good
    regions of the data and ignore the bad regions.

    """
    def __init__(self, noise_threshold=60, detect_noise=True):
        super().__init__()
        self.noise_threshold = noise_threshold
        self.detect_noise = detect_noise
        self.meta_segments = []

    def _detect_noisy_regions(self, data, segment_start, segment_end):
        noisy_regions = []
        freq_spectrum = np.abs(fft(data))
        freq_bins = np.fft.fftfreq(len(data))

        for i, amplitude in enumerate(freq_spectrum):
            if abs(freq_bins[i] - 60) < 1 and amplitude > self.noise_threshold:
                start, end = self._find_noise_segment(data, i, segment_start, segment_end)
                noisy_regions.append((start, end))

        return self._merge_overlapping_regions(noisy_regions)

    def parse(self, data, segment_start=0, segment_end=None):
        if segment_end is None:
            segment_end = len(data)

        noisy_regions = self._detect_noisy_regions(data, segment_start, segment_end)
        last_end = segment_start

        for start, end in noisy_regions:
            start = max(segment_start, start)
            end = min(segment_end, end)
            if start > last_end:
                self.meta_segments.append(MetaSegment(last_end, start, unique_features="clean"))
            self.meta_segments.append(MetaSegment(start, end, unique_features="noisy"))
            last_end = end

        if last_end < segment_end:
            self.meta_segments.append(MetaSegment(last_end, segment_end, unique_features="clean"))

        # Ensure all segments fit strictly within the parent range
        return [
            seg for seg in self.meta_segments
            if segment_start <= seg.start < seg.end <= segment_end
        ]


class IVCurveParser:
    """
    THIS CLASS IS NOT IMPLEMENTED YET! DO NOT USE.
    The class handles:

        1. Identifies a file where an IV curve is recorded based on the pattern.
    Here, the goal is to feed the voltage information to a module and have it search and do pattern matching to find one of
    these possibilities. .This can be a parser that accepts voltage array or compressed voltage as one of the inputs.


    """

    def __init__(self, voltage):
        #super().__init__()
        self.voltage = voltage

    def parse(self):
        """
        Searches for a pattern in the voltage data to identify IV.
        If none exists, the given datafile does not contain IV curves.
        If one or multiple IV curves are found, it returns start and
        end regions of each of them.
        Returns: list of indices: start and end for each deteced IV curve

        """
        patterns = self._defined_pattern()
        iv_regions = []

        for pattern_name, pattern in patterns.items():
            match_regions = self._find_pattern(self.voltage, pattern)
            if match_regions:
                for start, end in match_regions:
                    iv_regions.append((start, end))

        return patterns

    def _defined_pattern(self):
        """
        Defined common IV curve patterns
        """

        patterns = {
            "cyclical": [0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75, -1],
            "ramp": [0, 0.25, 0.5, 0.75, 1, 0, -0.25, -0.5, -0.75, -1],
            "flip_flop_zero": [0, 0.25, 0, -0.25, 0, 0.5, 0, -0.5, 0, 0.75, 0, -0.75, 0, 1, 0, -1, 0],
            "flip_flop_no_zero": [0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1]
        }
        return patterns

    def _find_pattern(self, pattern):
        """

        Match the pattern to find the occurrences of the pattern in
        the voltage array
        """

        pattern_length = len(pattern)
        match_region = []
        for i in range(len(self.voltage) - pattern_length + 1):
            # find a pattern comparing voltage array and the pattern
            if np.allclose(self.voltage[i:i + pattern_length], pattern, atol=0.001):
                match_region.append((i, i + pattern_length - 1))

        return match_region


class IVCurveAnalyzer(Parser): 
    """
    THIS CLASS IS NOT IMPLEMENTED YET. DO NOT USE.
    Check for at Least Two Voltage Steps: confirm that the segment has at least two voltage steps.

    Divide into Voltage Segments: separate data into  voltage segments.

    Call `IVAnalyzer` on Parent Segment: compute the mean and std of the current in each
    voltage segment.

    Return an Array of (voltage: (imean, istd)) pair:

    Also mean and std should be calulated in two ways
    """

    def __init__(self, current, parent_segment, sampling_frequency, method="simple"):
        super().__init__()
        self.current = current
        self.method = method
        self.parent_segment = parent_segment
        self.sampling_frequency = sampling_frequency

    def analyze(self):
        """
        Analyze each voltage step in a parent segment

        """
        # condition: at list two voltage steps are present
        if len(self.parent_segment.children) < 2:
            raise ValueError("The segment should have at least two voltage steps")
        result = {}
        for vstep in self.parent_segment.chindren:
            voltage = vstep.voltage
            current = vstep.current
            if self.method == "simple":
                imean, istd = self._simple_stats(current)
            elif self.method == "extra":
                imean, istd = self._extra_stats(current)
            else:
                raise ValueError("Invalid method. Choose 'simple' or 'extra'")
            result[voltage] = (imean, istd)

        return result

    def _simple_stats(self, current):
        return np.mean(current), np.tsd(current)

    def _extra_stats(self, current):
        """
        divide segment into n subsegments (default 10 or 0.1s, whichever is longer),
        sort based on imean of those subsegments, and extract the imean and istd
        from the subsegment with the highest imean magnitude (most positive or most negative).

        The goal of this is to ignore the regions where a biological pore is gating
        and only extract the segment where the pore looks most open.
        """
        subsegment_duration = 0.1
        current_data = current
        n_subsegments = int(max(10, subsegment_duration))



class AutoSquareParser(Parser):
    """
    Class for building a straightforward pipeline of data analysis
    to make it reusable with customized parameters
    """
    required_parent_attributes = ["current", "eff_sampling_freq", "voltage"]
    

    def __init__(self, threshold_baseline=0.7, expected_conductance=1.9,conductance_tolerance=1.2, wrap_padding:int=50, rules=[]):
        
        
        super().__init__()
        # self.voltage_range = voltage_range
        self.threshold = threshold_baseline
        self.wrap_padding = wrap_padding
        self.conductance_tolerance=conductance_tolerance
        self.expected_conductance = expected_conductance
        self.rules = rules

    def parse(self, current, eff_sampling_freq, voltage):
        sign=np.sign(voltage)
        if np.isclose(sign,0.0,atol=1e-3):
            sign=1
        # print(voltage, sign,len(current))
        results = []
        
        expected_baseline = voltage*sign*self.expected_conductance 
        current_positive=sign*current
        hist, edges = np.histogram(current_positive, bins=100, range=(expected_baseline*0.7,expected_baseline*1.4))
        centers = edges[:-1] + (edges[1] - edges[0]) * 0.5
        I0guess = centers[np.argmax(hist)]
        Ithresh = I0guess * self.threshold
        # if expected_baseline/1.2<I0guess <expected_baseline*1.2 :
        #     return []

        

        # lambda parser with dynamic rules
        lambda_parser = lambda_event_parser(
            threshold=Ithresh,
            rules=[
                      lambda event: 0 < event.mean < Ithresh*0.9,
                  ] + self.rules
        )
        events = lambda_parser.parse(current_positive)
        if len(events) == 0:
            return []
        ignored=[]
        for i in range(len(events)):
            
            
            
            
            """
            objective: find a baseline region to calculate the event-specific baseline
            cases: 
                if the event is the last in a set or the only event:
                    take the current between the end of the event and the end of the step
                otherwise, there is at least one event after this event, 
                    so take the current after the event up to the beginning of the next event as baseline.
                
            """
            if (events[i].start==0) or (events[i].start+events[i].duration)==current_positive.shape[0]:
                ignored.append(i)
                continue

            if i == len(events) - 1:
                bstart = events[i].start + events[i].duration

                bend = current_positive.shape[0]
            else:
                bstart = events[i].start + events[i].duration
                bend = events[i + 1].start
            # print(i, bstart,bend)
            baseline = current_positive[bstart:bend]

            
            #     return []
            if baseline.shape[0] > 0: 
                events[i].unique_features = {"baseline": np.median(baseline, axis=-1)}
            else:
                ignored.append(i) # this event does not have a baseline, meaning it ended at the end of the step. dump it
                continue

            if not(expected_baseline/self.conductance_tolerance<events[i].unique_features["baseline"] <expected_baseline*self.conductance_tolerance) :
                ignored.append(i)
                continue
            diff = np.diff(current_positive[events[i].start:events[i].start + events[i].duration])
            idxs = np.argwhere(diff > 0).ravel()
            if idxs.size == 0:
                # Skip this event if no positive difference is found 
                ignored.append(i)
                continue

            events[i].end=events[i].start+events[i].duration # assign temporary end of event to get the wrapping
            wstart = max(events[i].start -self.wrap_padding, 0) 
            if i>0:
                wstart = max(wstart,events[i-1].start+events[i-1].duration) #if previous event is too close, limit the wrap start
            wend = min(events[i].end + self.wrap_padding, current_positive.shape[0]) 
            if i<len(events)-1:
                wend = min(wend, events[i+1].start) #if next event is too close, limit the wrap end
            wstart = int(wstart) 
            wend = int(wend)

            events[i].unique_features["wrap"] = current_positive[wstart:wend]
            
            new_start = events[i].start + idxs[0]
            new_end = events[i].start + idxs[-1] + 1

            events[i].start = new_start
            events[i].end = new_end
            # events[i].duration=events[i].start-events[i].end
        # print(ignored)
        # print(len(events))
        # for event in events:
            # print(event.start,event.end)    
        events = [event for i, event in enumerate(events) if not (i in ignored)]
        # print(len(events))
        # print("survived:",[(event.start,event.end) for event in events])
        for event in events:
            assert hasattr(event,"unique_features"), f"Faulty event situation at {event.start},{event.end}"
            assert ("baseline" in event.unique_features) and "wrap" in event.unique_features,f"Event with no baseline or wrapping slipped through the parser at {event.start},{event.end}"
            #events[i].unique_features = {"baseline": np.median(baseline, axis=-1)}
        

        # Adjust the events by removing transitions

        
        for event in events:
            # event.duration = (event.end - event.start) / eff_sampling_freq

            # Unique features
            event.unique_features["baseline"]*=sign
            event.unique_features["mean"] = np.mean(current[event.start:event.end])
            event.unique_features["frac"] = 1 - (event.unique_features["mean"] / event.unique_features["baseline"])
            # event.unique_features[""] = event.end-event.start
            # event.unique_features["current"] = current[event.start:event.end]
            # event.unique_features["start"] = event.start
            # event.unique_features["end"]=event.end


            results.append((event.start, event.end, event.unique_features))

        return results
    



