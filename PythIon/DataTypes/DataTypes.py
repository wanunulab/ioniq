import numpy as np

from PythIon.DataTypes.CoreTypes import *
from PythIon.utils import Singleton
import datetime
import matplotlib.pyplot as plt
class SessionFileManager(MetaSegment,metaclass=Singleton):
    rank="root"
    def __init__(self)->None:
        super().__init__(self,end=2)
        self.unique_features["SessionStartTime"]=datetime.datetime.now()
    def add_child(self, child: AnySegment) -> None:
        self.children.append(child)
    def add_children(self, children: list[AnySegment]) -> None:
        self.children.extend(children)
    def _set_children_nocheck(self, children: list[AnySegment]) -> None:
        self.children.clear()
        self.children=children
        
        
        

class TraceFile(Segment):
    def __init__(self,current:np.ndarray,voltage=None,rank="file",parent=None,unique_features:dict={},metadata:dict={}):
        super().__init__(current)
        self.rank="file"
        self.parent=parent
        self.unique_features=unique_features
        self.voltage = voltage
        self.start=0
        self.end=len(self.current)
        self.metadata=metadata
        self.sampling_freq=self.unique_features.get("sampling_freq")
        self.t=np.arange(self.start,self.end)/self.sampling_freq
        if self.voltage is not None:
            self.add_children([MetaSegment(start,end,parent=self,rank="vstep",unique_features={"voltage":v}) for (start,end),v in voltage])
        if self.parent!= None:
            self.parent.add_child(self)
    def plot (self,rank,ax,downsample_per_rank,color_per_rank):
        pass


                
        
        
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
#         if (seconds and isinstance(self.start,float)) or (not seconds and isinstance(self.start,int)):
#             return(self.start,self.end)
#         if not seconds and isinstance(self.start,float):
#             return(int(self.start*self.sampling_freq),int(self.end*self.sampling_freq))
        



        