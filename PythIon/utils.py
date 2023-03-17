import numpy as np

class Singleton(type):
    def __init__(self,*args,**kwargs):
        self.__instance=None
        super().__init__(*args,**kwargs)
    def __call__(self,*args,**kwargs):
        if self.__instance is None:
            self.__instance=super().__call__(*args,**kwargs)
        return self.__instance
        
def split_voltage_steps(voltage:np.ndarray,n_remove=0,as_tuples=False):
    # Check if the current or voltage arrays are empty
    if not voltage.size:
        return []
    
    # Check if the n_remove argument is negative
    if n_remove < 0:
        raise ValueError("n_remove must be non-negative")
    # Find the indices at which the voltage level changes
    split_indices = np.where(voltage[:-1] != voltage[1:])[0] + 1
    
    # Add the start and end indices of the current array to the split indices
    split_indices = np.concatenate([[0], split_indices, [len(voltage)]])
    
    # # Calculate the start and end indices of the splits
    start_indices = split_indices[:-1] + n_remove
    end_indices = split_indices[1:]
    
    if np.any(end_indices <= start_indices):
        raise ValueError("n_remove is too large")
    
    if not as_tuples:
        return start_indices,end_indices
    else:
        return [(start_ind,end_ind) for start_ind,end_ind in zip(start_indices,end_indices)]