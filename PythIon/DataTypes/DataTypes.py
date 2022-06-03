import numpy as np

from PythIon.DataTypes.coretypes import *

class RoiSegment(MetaSegment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keys=kwargs.keys()
        if hasattr(self,'parent'):
            if hasattr(self.parent,'sampling_freq'):
                self.sampling_freq=self.parent.sampling_freq
    
    def get_bounds(self,seconds=True):
        if seconds and isinstance(self.start,int):
            return(self.start/self.sampling_freq,self.end/self.sampling_freq)
        if (seconds and isinstance(self.start,float)) or (not seconds and isinstance(self.start,int)):
            return(self.start,self.end)
        if not seconds and isinstance(self.start,float):
            return(int(self.start*self.sampling_freq),int(self.end*self.sampling_freq))
        



        