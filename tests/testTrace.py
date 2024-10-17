# import unittest
# import PythIon.DataTypes.CoreTypes as CT
# import numpy as np
#
# class SegmentTreeTestCase(unittest.TestCase):
#     def setUp(self):
#         self.n=10000
#         self.trace=np.random.randn(self.n)/100+2
#         self.topseg=CT.Segment(current=self.trace,start=0,end=10000)
#     def test_segment_children_addition(self):
#
#         child_full=CT.MetaSegment(start=0,end=10000,parent=self.topseg)
#         self.topseg.add_child(child_full)
#         self.assertEqual(self.topseg.slice,self.topseg.children[0].slice)
#         self.topseg.clear_children()
#         del child_full
#         children_partial=[]
#         for i in range (10):
#             children_partial.append(CT.MetaSegment(start=i*1000,end=(i+1)*1000,parent=self.topseg))
#         self.topseg.add_children(children_partial)
#         for child in self.topseg.children:
#             self.assertIn(child,children_partial)
#         bad_child=CT.MetaSegment(start=2000,end=3001,parent=self.topseg)
#         with self.assertRaises(AssertionError):
#             self.topseg.add_child(bad_child)
#     def test_performance_segment_children_addition(self):
#         self.trace=np.random.randn(10**7)/100+2
#         self.topseg=CT.Segment(current=self.trace,start=0,end=10**7)
#         import time
#
#         t1=time.time()
#         children_partial=[CT.MetaSegment(start=i,end=i+10,parent=self.topseg)for i in range(0,10**7,10)]
#         t2=time.time()
#         self.topseg.add_children(children_partial)
#         t3=time.time()
#         import matplotlib.pyplot as plt
#         plt.plot([0],[t2-t1])
#         plt.title(f"time to make {len(children_partial)} children: {t2-t1}s, time to add to parent: {t3-t2}s")
#         plt.waitforbuttonpress()
#     def test_DT(self):
#         import PythIon.DataTypes.DataTypes as dt
#         dt.SessionFileManager()
#
#         # self.topseg.children[0].slice
#
#
#
# class RawTraceSegmentTestCase(unittest.TestCase):
#     def setUp(self):
#         self.n=10000 # number of samples
#         self.trace=2+np.random.randn(self.n)/100
#         self.sampling_freq=200e3 #200kHz
#
#     def test_segment_properties(self):
#         self.segment=CT.Segment(current=self.trace, start=0,sampling_freq=self.sampling_freq)
#         self.assertEqual(self.segment.n,self.n)
#         self.assertTrue(np.array_equal(self.trace,self.segment.current),msg="generated trace does not match current in the segment object")
#         self.assertAlmostEqual(self.segment.mean,2,places=1)
#         self.assertEqual(self.segment.sampling_freq,self.sampling_freq)
#
#     def test_segment_deletion(self):
#         self.segment=CT.Segment(current=self.trace, start=0,sampling_freq=self.sampling_freq)
#         self.assertTrue(hasattr(self.segment,'current'))
#         self.segment.delete()
#         self.assertFalse(hasattr(self.segment,'current'))
#
#
#