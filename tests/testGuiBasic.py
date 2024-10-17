# import unittest
# import PythIon.MainApp as MainApp
# import numpy as np
# import os
# import threading
# class GuiDataLoad(unittest.TestCase):
#     def setUp(self):
#         # self.appthread=threading.Thread(target=MainApp.start)
#         # self.appthread.start()
#         self.qapp=MainApp.start()
#         self.app=MainApp.myapp
#         # self.app.showMaximized()
#         # self.trace=2+np.random.randn(self.n)/100
#         self.sampling_freq=200e3 #200kHz
#         self.app.pipeline_controls.connectTerminals(self.app.pipeline_controls['dataIn'],self.app.plot_trace_node['In'])
#     def test_file_load(self):
#         datafile="C:/Users/alito/EDR/R405M2_FBead_10pct_02/R405M2_FBead_10pct.edh"
#         self.app.load_data(datafile,os.path.dirname(datafile))
#         self.qapp.exec()
#     # def test_segment_deletion(self):
#     #     self.segment=CT.Segment(current=self.trace, start=0,sampling_freq=self.sampling_freq)
#     #     self.assertTrue(hasattr(self.segment,'current'))
#     #     self.segment.delete()
#     #     self.assertFalse(hasattr(self.segment,'current'))
#
#
#