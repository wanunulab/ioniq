import unittest

from SpikeAnalysis import *
import numpy as np
class TestDetectReversals(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    def test_detect_reversals_simple(self):
        voltage=[100]*10+[-100]*20+[100]*20+[-100]*10+[100]*10
        negatives=[20,10]
        positives=[10,20,10]
        starts,ends=detect_reversals(np.array(voltage),threshold=1,mark_above_threshold=True)
        
        for start,end,l in zip(starts,ends,positives):
            self.assertEqual(end-start,l)
        for start,end,l in zip(ends[:-1],starts[1:],negatives):
            self.assertEqual(end-start,l)
            
        starts,ends=detect_reversals(np.array(voltage),threshold=1,mark_above_threshold=False)
        
        for start,end,l in zip(starts,ends,negatives):
            self.assertEqual(end-start,l)
        positives=[20]
        for start,end,l in zip(ends[:-1],starts[1:],positives):
            self.assertEqual(end-start,l)
    def test_detect_reversals_blank(self):
        voltage=[100]*10+[200]*20+[100]*20+[200]*10+[100]*10
        negatives=[20,10]
        positives=[10,20,10]
        starts,ends=detect_reversals(np.array(voltage),threshold=1,mark_above_threshold=True)
        
        for start,end,l in zip(starts,ends,positives):
            self.assertEqual(end-start,l)
        for start,end,l in zip(ends[:-1],starts[1:],negatives):
            self.assertEqual(end-start,l)
            
        starts,ends=detect_reversals(np.array(voltage),threshold=1,mark_above_threshold=False)
        
        for start,end,l in zip(starts,ends,negatives):
            self.assertEqual(end-start,l)
        positives=[20]
        for start,end,l in zip(ends[:-1],starts[1:],positives):
            self.assertEqual(end-start,l)
            
    def test_detect_reversals_pad(self):
        p=1
        voltage=[100]*20+[-100]*20+[100]*20+[-100]*10+[100]*10
        negatives=[20-2*p,10-2*p]
        positives=[20+p,20+2*p,10+p]
        starts,ends=detect_reversals(np.array(voltage),threshold=1,mark_above_threshold=True,padding=p)
        
        for start,end,l in zip(starts,ends,positives):
            self.assertEqual(end-start,l)
        for start,end,l in zip(ends[:-1],starts[1:],negatives):
            self.assertEqual(end-start,l)
            
            
        p=6
        negatives=[20-2*p]
        positives=[20+p,20+10+10+p]
        starts,ends=detect_reversals(np.array(voltage),threshold=1,mark_above_threshold=True,padding=p)
        
        for start,end,l in zip(starts,ends,positives):
            self.assertEqual(end-start,l)
        for start,end,l in zip(ends[:-1],starts[1:],negatives):
            self.assertEqual(end-start,l)

class TestExperiment(unittest.TestCase):
    def setUp(self) -> None:
        self.current=np.arange(0,100)
        self.voltage=np.ones(self.current.shape[0])
        self.voltage[5:40]*=-1
        self.voltage[60:75]*=-1
        self.experiment=Experiment(self.current,self.voltage,250e3)
        return super().setUp()
        
    def test_auto_cut_by_voltage(self):
        self.experiment.auto_cut_by_voltage(threshold=0.1,remove_above_threshold=True,padding=0)
        expected=self.current[np.where(self.voltage<0.1)]
        
        self.assertTrue(np.all(self.experiment.current==expected))
            
