"""
Test module for io.py
"""

import pytest
import os
from ioniq.io import EDHReader


# Test EDH reader
@pytest.fixture
def path_to_file():
	return """/Users/dinaraboyko/grad_school/wanunu_lab/data/ \
			cytKWT/8_6_2024/4_channel_2MGdmCl_buffer_cytK_P6_106_CH003/ \
			4_channel_2MGdmCl_buffer_cytK_P6.edh"""


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
