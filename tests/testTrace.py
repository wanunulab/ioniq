import unittest
import PythIon.DataTypes.CoreTypes as CT
import numpy as np

class RawTraceSegmentTestCase(unittest.TestCase):
    def setUp(self):
        self.n=10000 # number of samples
        self.trace=2+np.random.randn(self.n)/100 
        self.sampling_freq=200e3 #200kHz
        
    def test_segment_properties(self):
        self.segment=CT.Segment(current=self.trace, start=0,sampling_freq=self.sampling_freq)
        self.assertEqual(self.segment.n,self.n)
        self.assertTrue(np.array_equal(self.trace,self.segment.current),msg="generated trace does not match current in the segment object")
        self.assertAlmostEqual(self.segment.mean,2,places=1)
        self.assertEqual(self.segment.sampling_freq,self.sampling_freq)
        
    def test_segment_deletion(self):
        self.segment.delete()
        self.assertIsNone(self.segment)
        