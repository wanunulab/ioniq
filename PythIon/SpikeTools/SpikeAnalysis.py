import numpy as np

def detect_reversals(voltage, threshold = 0.001, mark_above_threshold=True, padding:int=0):
    """
    Generates two arrays of start and end points to be cut out of the data 

    Arguments:
        voltage -- Voltage array given in V or mV

    Keyword Arguments:
        threshold -- threshold of detection (in the same units as voltage) (default: {0.001})
        mark_above_threshold -- mark regions where voltage is above the threshold (True) or below (False).  (default: {True})
        padding -- how many points before and after the reversal event must be included in the removal. (default: {0})
    """
    
    
    if mark_above_threshold:
        mask = np.where( voltage > threshold, 1, 0 ) # Find where the voltage is above the threshold, replace with 1's
    else:
        mask = np.where( voltage < threshold, 1, 0 )
    mask = np.diff( mask )                  # Find the edges, marking them with a 1 or -1 , by derivative
    starts=np.where(mask==1)[0]+1
    ends=np.where(mask==-1)[0]+1
    del mask
    if starts.shape[0]==0 and ends.shape[0]==0:
        return starts,ends
    if starts.shape[0]==0:
        starts=np.array([0])
    elif starts[0]>ends[0]:  # check if the start of the reversal is missed, add position 0 to the beginning  
        starts=np.concatenate(([0],starts))
    
    if ends.shape[0]==0:
        ends=np.array([voltage.shape[0]])    
    elif starts[-1]>ends[-1]: # check if the end of the reversal is missed, add end point of voltage to the end
        ends=np.concatenate((ends,[voltage.shape[0]]))

    padded_starts=starts-padding
    if padded_starts[0]<0:
        padded_starts[0]=0
    padded_ends=ends+padding
    if padded_ends[-1]>voltage.shape[0]:
        padded_ends[-1]=voltage.shape[0]
        
    overlaps=np.where(padded_starts[1:]<=padded_ends[:-1])[0]
    if overlaps.size>0:
        padded_starts=np.delete(padded_starts,overlaps+1)
        padded_ends=np.delete(padded_ends,overlaps)
    del overlaps
    return padded_starts,padded_ends

class Experiment():
    def __init__(self,current,voltage,sampling_freq) -> None:
        self.current=current
        self.voltage=voltage
        self.process=[]
        self.sampling_freq=sampling_freq
        pass
    
    def _cut_regions(self,starts,ends):
        regions=tuple(np.s_[start:end] for start,end in zip(starts,ends))
        regions=np.r_[regions]
        self.current=np.delete(self.current,regions)
        self.voltage=np.delete(self.voltage,regions)
    
    def auto_cut_by_voltage(self,threshold,remove_above_threshold=True,padding=0):
        starts,ends=detect_reversals(self.voltage,threshold=threshold,
                                     mark_above_threshold=remove_above_threshold,padding=padding)
        self._cut_regions(starts,ends)
        
    def parse(self,parser):
        self.events=parser.parse(self.current)
        
        
