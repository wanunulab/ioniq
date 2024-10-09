"""
Test module for datatypes.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ioniq.core import *
from ioniq.datatypes import SessionFileManager, TraceFile
import datetime
import numpy as np


def test_session_file_manager():
    """
    Test for SessionFileManager class
    """
    # instance of SessionFileManager
    instance = SessionFileManager()

    # Test initial state
    assert "SessionStartTime" in instance.unique_features
    assert isinstance(instance.unique_features["SessionStartTime"], datetime.datetime)


def test_session_file_manager_children():
    """
    Test SessionFileManager children management
    """
    #  instance
    session_manager = SessionFileManager()

    # create child segments
    child1 = MetaSegment(start=0, end=100, parent=session_manager)
    child2 = MetaSegment(start=101, end=200, parent=session_manager)

    # add single child
    session_manager.add_child(child1)
    assert len(session_manager.children) == 1
    assert session_manager.children[0] == child1

    # add multiple children
    session_manager.add_children([child2])
    assert len(session_manager.children) == 2
    assert child2 in session_manager.children

    # remove child
    session_manager._remove(child1)
    assert len(session_manager.children) == 1
    assert child1 not in session_manager.children
    # children without check
    session_manager._set_children_nocheck([child1, child2])
    assert len(session_manager.children) == 2
    assert session_manager.children == [child1, child2]


# Test for TraceFile class
def test_trace_file_init():
    """
    Test TraceFile class initialization
    """
    # set voltage step and current data
    voltage = [((0, 1000), 1.0), ((1000, 2000), 2.0)]
    current = np.random.randn(2000)

    trace_file = TraceFile(current=current, voltage=voltage, unique_features={"sampling_freq": 1000})

    # basic attributes
    assert trace_file.rank == "file"
    assert trace_file.start == 0
    assert trace_file.end == 2000
    assert trace_file.sampling_freq == 1000
    assert len(trace_file.time) == 2000
    assert isinstance(trace_file.time, np.ndarray)

    # Check child voltage steps
    assert len(trace_file.children) == 2
    assert trace_file.children[0].rank == "vstep"
    assert trace_file.children[0].unique_features["voltage"] == 1.0
    assert trace_file.children[0].start == 0
    assert trace_file.children[0].end == 1000

