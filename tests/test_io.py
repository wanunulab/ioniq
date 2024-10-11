"""
Test module for io.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))
import pytest
import os
from ioniq.io import EDHReader


# Test EDH reader
@pytest.fixture
def path_to_file():
	return "data/8e7_80n01M1_5pctSorbitol_IV/8e7_80n01M1_5pctSorbitol_IV.edh"


def test_edh_reader(path_to_file):
	"""Test EDH reader"""
	edh_reader = EDHReader()
	assert os.path.exists(path_to_file), f"EDH file not found at {path_to_file}"

	# Read the file
	metadata, current, voltage = edh_reader.read(path_to_file)
	# Check keys in metadata
	assert "EDH Version" in metadata
	# Check the current is read and it is not empty
	assert current is not None
	assert len(current) > 0
	# Check the voltage is read and it is not empty
	assert voltage is not None
	assert len(voltage) > 0
