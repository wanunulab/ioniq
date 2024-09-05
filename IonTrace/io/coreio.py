import os
import numpy as np
import pyabf
import glob
from PythIon.utils import split_voltage_steps
class AbstractFileReader(object):
    
    ext="___" # replace with the appropriate file extension in subclass, such as ".abf", ".edh", ".mat", etc.
    accepted_keywords=[] # replace with a list of accepted keyword arguments passed to the _read() function in every subclass.
    
    # Values to scale voltage and current by to bring them to SI units (A and V). Change these in subclasses to match the idata scale
    current_multiplier:float = 1.0
    voltage_multiplier:float = 1.0
    
    def __init__(self):
        self.filename="UNDEFINED"
    
    def read(self,filename,**kwargs):
        """
        Read a datafile or series of files. Files are identified according to their extension.
        Data formats that come with a header file must be referred to by the header file.
        If inheriting from the AbstractFileReader class, do not override this function; instead, create a custom _read method.

        :param filename: file name or list of file names, typically given as a string or PathLike object.
        :type filename: str or os.PathLike or list[str] or list[os.PathLike]
        :param **kwargs: keyword arguments passed directly to the file reader class that matches the data format.
        :return: [metadata, current, etc.. ]. If the input "filename" is a list, this function returns a generator object that yields the output of _read() for every file in the input list.
        :rtype: tuple[dict,np.ndarray [,np.ndarray or tuple[slice,np.float32]]]
        """
        for key in kwargs.keys():
            if key not in self.accepted_keywords:
                raise(TypeError(f"{self.__class__}.read() got an unexpected argument: {key}"))
        if type(filename) is list:
            self.kwargs=kwargs
            for fname in filename:
                assert(os.path.splitext(fname)[-1].lower()==self.ext.lower())
            return (self._read(fname,**kwargs) for fname in filename)

        else:
            assert(os.path.splitext(filename)[-1].lower()==self.ext.lower())
            self.kwargs=kwargs
            return self._read(filename,**kwargs)
        
    def _read(self,filename,**kwargs):
        pass #rewrite this function in inherited classes to process the data
    
    def __repr__ (self):
        print(self.filename)
        
class EDHReader(AbstractFileReader):
    ext=".edh" 
    accepted_keywords=["voltage_compress","n_remove","downsample","prefilter"]
    
    current_multiplier=1e-9 #current is stored in nA in the datafile
    voltage_multiplier=1e-3 #voltage is stored in mV in the datafile
    def __init__(self):
        super().__init__()
        
    def _read(self,filename,**kwargs):
        from pyqtgraph import siEval
        filename=os.path.abspath(filename)
        direc=os.path.dirname(filename)
        metadata={}
        
        with open(filename,'r') as headerfile:
            for line in headerfile:
                lsplit=line.split(":")
                match lsplit:
                    case ["EDH Version" | "Channels" | "Oversampling x4" | "Active channels",*val]:
                        metadata[lsplit[0]]="".join(val).strip()
                    case ["Sampling frequency (SR)"|"Range",*val]:
                        metadata[lsplit[0]]=siEval("".join(val).strip())
                    case ["Final Bandwidth",*val]:
                        metadata["Final Bandwidth"]=metadata["Sampling frequency (SR)"]/int(val[0].strip().split()[0].split("/")[1])
                    case ["Acquisition start time",*val]:
                        metadata["Acquisition start time"]=" ".join(line.split(" ")[-2:])
                    case _:
                        pass
        # print(metadata)
        # if multichannel:
        #     active_channels= list(map(int,metadata["Active channels"].split()))
        #     active_channels=[str(x-1) for x in active_channels]
        #     core_fname=os.path.splitext(os.path.split(filename)[-1])[0]
        #     for channel_name in active_channels:
        #         file_list_abf=glob.glob(f"{core_fname}_CH00{channel_name}_*.abf",root_dir=direc)
                
        file_list_abf=glob.glob("*.abf",root_dir=direc)
        
        if len(file_list_abf)>0:
            abf_buffers=tuple(map(pyabf.ABF,[os.path.join(direc,file) for file in file_list_abf]))
            # list(map())
            current=np.concatenate([buffer.data[0] for buffer in abf_buffers],axis=0,dtype=np.float32)
            
            voltage=np.concatenate([buffer.data[-1] for buffer in abf_buffers],axis=0,dtype=np.float32)
            metadata["DataFiles"]=file_list_abf
            metadata["StorageFormat"]=".abf"
        else:
            file_list_dat=glob.glob("*.dat",root_dir=direc)
            if len(file_list_dat)==0:
                raise(FileExistsError("No associated data files (*.abf or *.dat) found."))
            data=np.concatenate([np.fromfile(os.path.join(direc,file),dtype="float32") for file in file_list_dat])
            data=data.reshape((int(metadata["Active channels"])+1,-1),order="F")
            current=data[0]
            voltage=data[-1]
            metadata["DataFiles"]=file_list_dat
            metadata["StorageFormat"]=".dat"
        assert(current.shape==voltage.shape)
        metadata["HeaderFile"]=filename
        #Scale the current and voltage arrays to SI units
        if kwargs.get("prefilter",None):
            prefilter=kwargs.get("prefilter")
            assert callable(prefilter)
            prefilter(current)
        
        current*=self.current_multiplier
        voltage*=self.voltage_multiplier
        if kwargs.get("downsample",None):
            downsample_factor=kwargs.get("downsample")
            assert type(downsample_factor) is int,f"non-integer downsampling factor not supported: {type(downsample_factor)}, {downsample_factor}"
            if downsample_factor>1:
                _current=current[::downsample_factor].copy()
                _voltage=voltage[::downsample_factor].copy()
                del current,voltage
                current,voltage=_current,_voltage
                metadata["downsample"]=downsample_factor
                metadata["eff_sampling_freq"]=metadata["Sampling frequency (SR)"]/downsample_factor
            
            
        if kwargs.get("voltage_compress",False) == True:
            n_remove=kwargs.get("n_remove",0)
            voltage_splits=split_voltage_steps(voltage,as_tuples=True,n_remove=n_remove)
            voltage_points=[(sl,voltage[sl[0]]) for sl in voltage_splits]
            del voltage
            return metadata,current,voltage_points
        
        return metadata,current,voltage
            
            
        
    
    
if __name__ == "__main__":
    print(EDHReader.ext)
    e=EDHReader()
    meta,current,voltage=e.read("C:/Users/alito/EDR/R506M2_FSBSATBead/R506M2_FSBSATBead.edh",voltage_compress=True)
    import matplotlib.pyplot as plt
    # plt.plot(current[::100])
    # plt.waitforbuttonpress()
    # e.read("C:/Users/alito/EDR/Q402m1_SBead/Q402m1_SBead.edh")
    
    
    