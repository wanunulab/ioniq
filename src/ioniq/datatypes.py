#!/usr/bin/env python
"""
DataTypes module
"""

import datetime
import numpy as np
from ioniq.core import MetaSegment, AnySegment, Segment
from ioniq.utils import Singleton
from ioniq.setup_log import json_logger
import uuid


class SessionFileManager(MetaSegment, metaclass=Singleton):
    """
    Class inherits rom MetaSegment and uses the Singleton
    to create only one instance of the class.
    """
    rank = "root"  # set the rank to "root"

    @json_logger.log
    def __init__(self) -> None:
        """
        Initialize methods
        """
        super().__init__(self, end=2)  # end = 2: Segment end at position 2???
        self.unique_features["SessionStartTime"] = datetime.datetime.now()
        self.affector_table = {}

    # make log management table. contains everything that has happened in this session

    def register_affector(self, affector):

        # generate a uuid
        new_uuid = str(uuid.uuid4())
        try:
            affector_repr = repr(affector)
        except Exception:
            affector_repr = "<unrepresentable_repr>"
        entry = {
            "class": affector.__class__.__name__,
            "signature": affector_repr,
            "timestamp": datetime.datetime.now().isoformat()}

        self.affector_table[new_uuid] = entry  # entry here is the the entry to log structure(
        json_logger._log_to_json({
            "action": "register_affector",
            "uuid": new_uuid,
            "entry": entry
        })
        return new_uuid


    def add_child(self, child: AnySegment) -> None:
        """
        Add a single child
        """
        self.children.append(child)

    def add_children(self, children: list[AnySegment]) -> None:
        """
        Add multiple children
        """
        self.children.extend(children)

    def _set_children_nocheck(self, children: list[AnySegment]) -> None:
        """
        Clear all children and add new without check
        """
        self.children.clear()
        self.children = children

    def _remove(self, child):
        """
        Remove a child from the children list
        """
        if child in self.children:
            self.children.remove(child)


class TraceFile(Segment):
    """
    Class inherits from Segment and represents the file with the current dara and voltage
    """

    @json_logger.log
    def __init__(self, current: np.ndarray, voltage=None, rank="file", parent=None,
                 unique_features: dict = {}, metadata: dict = {}):
        """
        "vstep" = Voltage step

        """
        super().__init__(current)
        self.rank = "file"
        self.parent = parent
        self.unique_features = unique_features
        self.voltage = voltage
        self.start = 0
        self.end = len(self.current)
        self.metadata = metadata
        self.uuid = None
        self.sampling_freq = self.unique_features.get("sampling_freq")
        self.time = np.arange(self.start, self.end) / self.sampling_freq

        # If voltage data exists, create MetaSegment instance for each voltage step
        if self.voltage is not None:
            self.add_children([MetaSegment(start, end, parent=self, rank="vstep",
                                           unique_features={"voltage": v})
                               for (start, end), v in voltage])
        if self.parent is not None:
            self.parent.add_child(self)

    def plot(self, rank, axes, downsample_per_rank, color_per_rank):
        """
        Method for plotting
        """
        pass

    def delete(self):
        """
        Delete the current object, removing it from its parent

        """
        try:
            if self.parent is not None:
                self.parent._remove(self)
        except Exception as error:
            print(error)
        super().delete()


# class RoiSegment(MetaSegment):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.keys=kwargs.keys()
#         if hasattr(self,'parent'):
#             if hasattr(self.parent,'sampling_freq'):
#                 self.sampling_freq=self.parent.sampling_freq

#     def get_bounds(self,seconds=True):
#         if seconds and isinstance(self.start,int):
#             return(self.start/self.sampling_freq,self.end/self.sampling_freq)
#         if (seconds and isinstance(self.start,float)) or
#             (not seconds and isinstance(self.start,int)):
#             return(self.start,self.end)
#         if not seconds and isinstance(self.start,float):
#             return(int(self.start*self.sampling_freq),int(self.end*self.sampling_freq))
