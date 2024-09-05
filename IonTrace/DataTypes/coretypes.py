#
#			adaptation from from "core.py" by Jacob Scheriber
#           https://github.com/jmschrei/PyPore
# This holds the core data types which may be abstracted in many
# different applications.

import numpy as np
import re
import json
from contextlib import contextmanager
from itertools import chain
from typing import Type,TypeVar
from functools import cached_property
# from PythIon.parsers.parsers import AnyParser
AnySegment=TypeVar("AnySegment",bound="AbstractSegmentTree")
class AbstractSegmentTree(object):
    __slots__=('parent','children','start','end','rank','__dict__')
    def __init__(self) -> None:
        self.parent: AnySegment | None = None
        self.children: list[AnySegment]=[]
        
        # # self._slice_relative: Type(np.s_)= np.s_[::]
        self.start: int|None =0
        self.end: int|None =None
        self.rank: str=None
        pass
    def parse(self,parser,newrank:str,at_child_rank:str|None=None, **kwargs)->bool:
        """
        Parses the data in the segment or at a particular rank of child segemts into children segments with a new rank. 

        :param parser: a Parser subclass instance 
        :type parser: AnyParser
        :param newrank: The rank string to be assigned to the newly found children
        :type newrank: str
        :param at_child_rank: Determines whether to traverse the children tree down to a given rank and apply the parser to those children or (if None) parse this instance, defaults to None
        :type at_child_rank: str | None, optional
        :return: True if no errors were encountered, else false 
        :rtype: bool
        """
        try:
            required_parent_attributes=parser.get_required_parent_attributes()
            def _parse_one(target):
                attributes=dict([(attr_name, target.get_feature(attr_name)) for attr_name in required_parent_attributes])
                # if "sampling_freq" in required_parent_attributes:
                    # attributes["sampling_freq"]=self.climb_to_rank('file').unique_features["sampling_freq"]
                parser_results=parser.parse(**attributes,**kwargs)
                children = [MetaSegment(start+self.start,end+self.start,parent=target,rank=newrank,unique_features=unique_features)
                            for start,end,unique_features in parser_results]
                target.clear_children()
                target.add_children(children)
                
            if at_child_rank is None or at_child_rank == self.rank:
                _parse_one(self)
            else:
                targets=self.traverse_to_rank(at_child_rank)
                for target in targets:
                    _parse_one(target)
        except Exception as e:
            raise e
        
        return True
    
    def get_feature(self,name:str):
        if name in self.unique_features.keys():
            return self.unique_features[name]
        elif hasattr(self,name):
            return getattr(self,name)
        elif self.parent!= None:
            return self.parent.get_feature(name)
        else: 
            return None
        
    @cached_property
    def relative_start(self) -> int:
        if self.parent is not None:
            return self.start-self.parent.start
        return self.start
    @cached_property
    def relative_end(self)->int:
        if self.parent is not None:
            return self.end - self.parent.start
        return self.end
    @cached_property
    def slice(self):
        return np.s_[self.start:self.end]
    @cached_property
    def relative_slice(self):
        return np.s_[self.relative_start:self.relative_end]
    @cached_property
    def n(self)-> int:
        return self.end-self.start
    
    def get_top_parent(self) -> AnySegment:
        if self.parent is not None:
            return self.parent.get_top_parent(self)
        else:
            return self
        
    def climb_to_rank(self,rank:str) -> AnySegment | None:
        if self.rank == rank:
            return self
        elif self.parent is not None:
            return self.parent.climb_to_rank(rank)
        else:
            return None
        
    def add_child(self,child: AnySegment) -> None:
        self.add_children([child])
            
    def add_children(self,children: list[AnySegment]) -> None:
        if self.start is not None and self.end is not None:
            assert (all( [self.start <= child.start < child.end <= self.end for child in children]))
        else:
            assert (all( [child.n>0 for child in children]))
        temp_children=sorted(self.children+children,key=lambda x: (x.start,x.end))
        assert (all( [child0.end<=child1.start for child0,child1 in zip(temp_children[:-1],temp_children[1:])]))
        self._set_children_nocheck(temp_children)
        
    def _set_children_nocheck(self,children: list[AnySegment]) -> None:
        self.children=children
        
    def clear_children(self):
        self.children.clear()
        return
    

    def traverse_to_rank(self,rank:str):
        if self.rank==rank:
            return [self]
        else:
            if self.children==[]:
                return []
            else: 
                return list(chain(*[child.traverse_to_rank(rank) for child in self.children]))


# class ParsableMixin:
#     def parse(self,parser,newrank:str,**kwargs)->bool:
#         required_parent_attributes=parser.get_required_parent_attributes()
#         attributes=dict([(attr_name, getattr(self,attr_name,None)) for attr_name in required_parent_attributes])
#         self.clear_children()
#         parser_results=parser.parse(**attributes,**kwargs)
#         children = [MetaSegment(start,end,parent=self,rank=newrank,unique_features=unique_features)
#                     for start,end,unique_features in parser_results]
#         self.add_children(children)
            
  
class MetaSegment(AbstractSegmentTree):
    '''
    The metadata on an abstract segment of ionic current. All information about a segment can be 
    loaded, without the expectation of the array of floats.
    '''
    __slots__=("unique_features")
    def __init__(self, start:int, end:int,
                 parent:AnySegment|None=None,rank:str|None = None,
                 unique_features: dict | None={},**kwargs):
        super().__init__()
        # pointers to lower level segments (sub-segments)
        # self.slice=np.s_[::]
        # self.slice_relative=np.s_[::]
        self.start=start
        self.end=end
        self.parent=parent
        self.unique_features=unique_features
        self.rank=rank
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        # If current is passed in, get metadata directly from it, then remove
        # # the reference to that array.
        # if hasattr(self, "current"):
        #     self.n = len(self.current)
        #     self.mean = np.mean(self.current)
        #     self.std = np.std(self.current)
        #     self.min = np.min(self.current)
        #     self.max = np.max(self.current)
        #     del self.current

        # Fill in start, end, and duration given that you only have two of them.
        # if hasattr(self, "start") and hasattr(self, "end") and not hasattr(self, "duration"):
        #     self.duration = self.end - self.start
        # elif hasattr(self, "start") and hasattr(self, "duration") and not hasattr(self, "end"):
        #     self.end = self.start + self.duration
        # elif hasattr(self, "end") and hasattr(self, "duration") and not hasattr(self, "start"):
        #     self.start = self.end - self.duration
    @property
    def current(self):
        try:
            c = self.climb_to_rank('file').current[self.start:self.end]
            return c
        except:
            return None
    @property
    def mean(self):
        return np.mean(self.current)

    @property
    def std(self):
        return np.std(self.current)

    @property
    def min(self):
        return np.min(self.current)

    @property
    def max(self):
        return np.max(self.current)
    
    @property
    def t(self):
        return self.climb_to_rank("file").t[self.start:self.end]
    
    @property
    def duration(self):
        return self.t[-1]-self.t[0]
    
    def __repr__(self):
        '''
        The representation is a JSON.
        '''

        return self.to_json()

    def __len__(self):
        '''
        The length of the metasegment is the length of the ionic current it
        is representing.
        '''

        return self.n
    def delete(self):
        '''
        Delete itself. There are no arrays with which to delete references for.
        '''

        del self
    
    def to_meta(self):
        '''
        Kept to allow for error handling, but since it's already a metasegment
        it won't actually do anything.
        '''

        pass

    def to_dict(self):
        '''
        Return a dict representation of the metadata, usually used prior to
        converting the dict to a JSON.
        '''
        if hasattr (self,'keys'):
            keys=self.keys
        else:
            keys = ['mean', 'std', 'min', 'max', 'start', 'end', 'duration']
        d = {i: getattr(self, i) for i in keys if hasattr(self, i)}
        d['name'] = self.__class__.__name__
        return d

    def to_json(self, filename=None):
        '''
        Return a JSON representation of this, by reporting the important
        metadata.
        '''

        _json = json.dumps(self.to_dict(), indent=4, separators=(',', ' : '))
        if filename:
            with open(filename, 'w') as outfile:
                outfile.write(_json)
        return _json

    @classmethod
    def from_json(self, filename=None, json=None):
        '''
        Read in a metasegment from a JSON and return a metasegment object. 
        Either pass in a file which has a segment stored, or an actual JSON 
        object.
        '''

        assert filename or json and not (filename and json)
        import re

        if filename:
            with open(filename, 'r') as infile:
                json = ''.join([line for line in infile])

        words = re.findall(r"\[[\w'.-]+\]|[\w'.-]+", json)
        attrs = {words[i]: words[i+1] for i in range(0, len(words), 2)}

        return MetaSegment(**attrs)



class Segment(AbstractSegmentTree):
    '''
    A segment of ionic current, and methods relevant for collecting metadata. The ionic current is
    expected to be passed as a numpy array of floats. Metadata methods (mean, std..) are decorated 
    as properties to reduce overall computational time, making them calculated on the fly rather 
    than during analysis.
    '''

    def __init__(self, current, **kwargs):
        '''
        The segment must have a list of ionic current, of which it stores some statistics about. 
        It may also take in as many keyword arguments as needed, such as start time or duration 
        if already known. Cannot override statistical measurements. 
        '''
        super().__init__()
        self.current = current
        for key, value in kwargs.items():
            with ignored(AttributeError):
                setattr(self, key, value)    
    
    
    def __repr__(self):
        '''
        The string representation of this object is the JSON.
        '''
        return self.to_json()

    def __len__(self):
        '''
        The length of a segment is the length of the underlying ionic current
        array.
        '''

        return self.n

    def to_dict(self):
        '''
        Return a dict representation of the metadata, usually used prior to
        converting the dict to a JSON.
        '''
        if hasattr(self, 'keys'):
            keys = self.keys
        else:
            keys = ['mean', 'std', 'min', 'max', 'start', 'end', 'duration']
        d = {i: getattr(self, i) for i in keys if hasattr(self, i)}
        d['name'] = self.__class__.__name__
        return d

    def to_json(self, filename=None):
        '''
        Return a JSON representation of this, by reporting the important
        metadata.
        '''

        _json = json.dumps(self.to_dict(), indent=4, separators=(',', ' : '))
        if filename:
            with open(filename, 'w') as outfile:
                outfile.write(_json)
        return _json

    def to_meta(self):
        '''
        Convert from a segment to a 'metasegment', which stores only metadata
        about the segment and not the full array of ionic current.
        '''

        for key in ['mean', 'std', 'min', 'max', 'end', 'start', 'duration']:
            with ignored(KeyError, AttributeError):
                self.__dict__[key] = getattr(self, key)
        del self.current

        self.__class__ = type("MetaSegment", (MetaSegment, ), self.__dict__)

    def delete(self):
        '''
        Deleting this segment requires deleting its reference to the ionic
        current array, and then deleting itself. 
        '''

        with ignored(AttributeError):
            del self.current

        del self

    def scale(self, sampling_freq):
        '''
        Rescale all of the values to go from samples to seconds.
        '''

        with ignored(AttributeError):
            self.start /= sampling_freq
            self.end /= sampling_freq
            self.duration /= sampling_freq

    @property
    def mean(self):
        return np.mean(self.current)
    
    @property
    def std(self):
        return np.std(self.current)

    @property
    def min(self):
        return np.min(self.current)

    @property
    def max(self):
        return np.max(self.current)

    @property
    def n(self):
        return len(self.current)

    @classmethod
    def from_json(self, filename=None, json=None):
        '''
        Read in a segment from a JSON and return a metasegment object. Either
        pass in a file which has a segment stored, or an actual JSON object.
        '''

        assert filename or json and not (filename and json)
        import re

        if filename:
            with open(filename, 'r') as infile:
                json = ''.join([line for line in infile])

        if 'current' not in json:
            return MetaSegment.from_json(json=json)

        words = re.findall(r"\[[\w\s'.-]+\]|[\w'.-]+", json)
        attrs = {words[i]: words[i+1] for i in range(0, len(words), 2)}

        current = np.array([float(x) for x in attrs['current'][1:-1].split()])
        del attrs['current']

        return Segment(current, **attrs)


@contextmanager
def ignored(*exceptions):
    '''
    Replace the "try, except: pass" paradigm by replacing those three lines with a single line.
    Taken from the latest 3.4 python update push by Raymond Hettinger, see:
    http://hg.python.org/cpython/rev/406b47c64480
    '''
    try:
        yield
    except exceptions:
        pass
