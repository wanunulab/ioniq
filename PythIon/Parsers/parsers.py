#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@soe.ucsc.com
# parsers.py 
# 
# This program will read in an abf file using read_abf.py and
# pull out the events, saving them as text files.


import sys
from itertools import tee,chain
import re

from typing import Type,TypeVar

import time
import numpy as np
# try:
#     from PyQt4 import QtGui as Qt
#     from PyQt4 import QtCore as Qc
# except:
    # pass
# from PythIon.DataTypes.CoreTypes import *

# import pyximport
# pyximport.install( setup_args={'include_dirs':np.get_include()})

from PythIon.parsers.cparsers import FastStatSplit
from PythIon.DataTypes.CoreTypes import Segment, MetaSegment
AnyParser=TypeVar("AnyParser",bound="Parser")

import json




#########################################
# EVENT PARSERS
#########################################

class Parser( object ):
    # override this with the required attributes to get from the parent segment that the parser needs
    required_parent_attributes:list[str]=['current']  
    def __init__( self ):
        pass
    
    @classmethod
    def get_name(cls):
        return cls.__name__
    
    @classmethod
    def get_init_params(cls):
        return getattr(cls,"_params",None)
    
    @classmethod
    def get_process_inputs(cls):
        return getattr(cls,"_parse_inputs",None)
    
    @classmethod
    def get_process_outputs(cls):
        return getattr(cls,"_parse_outputs",None)
    
    # @classmethod
    def get_required_parent_attributes(self):
        return getattr(self,"required_parent_attributes",[])
            
    def __repr__( self ):
        ''' Returns a representation of the parser in the form of all arguments. '''
        return self.to_json()

    def to_dict( self ):
        d = { key: val for key, val in list(self.__dict__.items()) if key != 'param_dict'
                                                             if type(val) in (int, float)
                                                                    or ('Qt' not in repr(val) )
                                                                    and 'lambda' not in repr(val) }
        d['name'] = self.__class__.__name__
        return d

    def to_json( self, filename=False ):
        _json = json.dumps( self.to_dict(), indent=4, separators=( ',', ' : ' ) )
        if filename:
            with open( filename, 'r' ) as out:
                out.write( _json )
        return _json

    def parse( self, **kwargs ):
        '''  Takes in a current segment, and returns a list of segment objects. ''' #TODO: change this description
        current=kwargs['current']
        return [(0,current.shape[0],{})]
        # return [ Segment( current=current, start=0, duration=current.shape[0]/100000 ) ]

    # def GUI( self ):
    #     '''
    #     A generic GUI built using PyQt4 based off the arguments presented in upon
    #     initialization. Requires PyQt4 to use, but the parser class does not require
    #     PyQt4 to run.
    #     '''
    #     grid = Qt.QGridLayout()
    #     param_dict = {}
    #     for i, (key, val) in enumerate( self.__dict__.items() ):
    #         param_dict[key] = Qt.QLineEdit()
    #         param_dict[key].setText( str(val) )
    #         grid.addWidget( Qt.QLabel(key), i, 0 )
    #         grid.addWidget( param_dict[key], i, 1 )
    #     self.param_dict = param_dict
    #     return grid

    def set_params( self,param_dict ):
        '''
        Updates each paramater presented in the GUI to the value input in the lineEdit
        corresponding to that value.
        '''
        try:
            # for key, lineEdit in list(self.param_dict.items()):
            #     val = lineEdit.text()
            #     if '.' in val:
            #         setattr( self, key, float( val ) )
            #         continue
            #     for i, letter in enumerate(val):
            #         if not letter.isdigit():
            #             setattr( self, key, str( val ) )
            #             continue
            #         if i == len(val):
            #             setattr( self, key, int( val ) )
            self.param_dict=param_dict
            for key,value in self.param_dict.items():
                setattr(self,key,value)
        except:
            pass

    @classmethod
    def from_json( self, _json ):
        """ Critical: Not implemented yet, figure out how to make this work in pythion"""
        if _json.endswith(".json"):
            with open( _json, 'r' ) as infile:
                _json = ''.join(line for line in infile)

        d = json.loads( _json )
        name = d['name']
        del d['name']

        return None #getattr( PyPore.parsers, name )( **d )

class SpikeParser(Parser):
    required_parent_attributes=["current","sampling_freq","mean","start","std"]
    
    _fractionable_attrs=['height','threshold','prominence']
    _time_attrs=['distance','prominence','width','wlen','plateau_size']
    _other_attrs=['rel_height']
    def __init__(self,height=None,threshold=None,distance=None,prominence=None,prominence_snr=None,width=None,wlen=None,rel_height:float=0.5,plateu_size=None,fractional:bool=True) -> None:
        self._height=height
        self._threshold=threshold
        self._distance=distance
        self._prominence=prominence
        self._prominence_snr=prominence_snr
        self._width=width
        self._wlen = wlen
        self._rel_height=rel_height
        self._plateau_size=plateu_size
        self._fractional=fractional
        
        
    def _calculate_absolute_from_fractionals(self,mean):
        pass
    def parse(self,current,sampling_freq,mean,std,start):
        from scipy import signal
        abs_start=start
        for key in self._time_attrs:
            _key='_'+key
            attr=getattr(self,_key,None)
            if attr !=None:
                if type(attr) in [tuple,list]:
                        # print(sampling_freq)
                        setattr(self,key,
                                (sampling_freq*attr[0] if attr[0] is not None else None,
                                 sampling_freq*attr[1] if attr[0] is not None else None))
                else:
                    setattr(self,key,sampling_freq*attr)
            else:
                setattr(self,key,attr)
        
        
        for key in self._fractionable_attrs:
            _key='_'+key
            attr=getattr(self,_key,None)
            if self._fractional:
                if attr !=None:
                    if type(attr) in [tuple,list]:
                            setattr(self,key,
                                    (mean*attr[0] if attr[0] is not None else None,
                                    mean*attr[1] if attr[0] is not None else None))
                    else:
                        setattr(self,key,mean*attr)
                else:
                    setattr(self,key,attr)
            else:
                setattr(self,key,attr)
                
        if self._prominence_snr is not None and self._fractional:
            if type(self._prominence_snr) in [tuple,list]:
                level=std/np.sqrt(np.abs(mean))
                self.prominence=(self._prominence_snr[0]*level if self._prominence_snr[0] is not None else None,
                                  self._prominence_snr[1]*np.abs(mean) if self._prominence_snr[1] is not None else self.prominence[1] if type(self.prominence) in [list,tuple] else None)
                
        for key in self._other_attrs:
            setattr(self,key,getattr(self,"_"+key))
        
        all_keys=self._fractionable_attrs+self._time_attrs+self._other_attrs
        find_peaks_args=dict([(key,getattr(self,key)) for key in all_keys])
        peaks,props=signal.find_peaks(current,**find_peaks_args)
        dt = np.array(0)
        dt=np.append(dt,np.diff(peaks)/sampling_freq)
        for i,peak in enumerate(peaks):
            event_start=round(props["left_ips"][i])
            event_end=round(props["right_ips"][i])
            unique_features={
                "idx_rel":peak,
                "idx_abs":abs_start+peak,
                "peak_val":current[peak],
                "deli":props["prominences"][i],
                "dwell":props["widths"][i]*1e6/sampling_freq,
                "dt":dt[i],
                "frac":props["prominences"][i]/mean}
            yield event_start,event_end,unique_features
        return
            
        
        

class MemoryParse( object):
    '''
    A parser based on being fed previous split points, and splitting a raw file based
    those splits. Used predominately when loading previous split points from the 
    database cache, to reconstruct a parsed file from "memory.""
    '''
    def __init__( self, starts, ends ):
        self.starts = starts
        self.ends = ends
    def parse( self, current ):
        return [ Segment( current=np.array(current[int(s):int(e)], copy=True),
                          start=s,
                          duration=(e-s) ) for s, e in zip(self.starts, self.ends)]

class lambda_event_parser( Parser ):
    '''
    A simple rule-based parser which defines events as a sequential series of points which are below a 
    certain threshold, then filtered based on other critereon such as total time or minimum current.
    Rules can be passed in at initiation, or set later, but must be a lambda function takes in a PreEvent
    object and performs some boolean operation. 
    '''
    
    def __init__( self, threshold=90, rules=None ):
        self.threshold = threshold
        self.rules = rules or [ lambda event: event.duration > 100000,
                                lambda event: event.min > -0.5,
                                lambda event: event.max < self.threshold ]
    def _lambda_select( self, events ):
        '''
        From all of the events, filter based on whatever set of rules has been initiated with.
        ''' 
        return [ event for event in events if np.all( [ rule( event ) for rule in self.rules ] ) ]
    
    def parse( self, current ):
        '''
        Perform a large capture of events by creating a boolean mask for when the current is below a threshold,
        then detecting the edges in those masks, and using the edges to partitition the sample. The events are
        then filtered before being returned. 
        '''
        mask = np.where( current < self.threshold, 1, 0 ) # Find where the current is below a threshold, replace with 1's
        mask = np.abs( np.diff( mask ) )                  # Find the edges, marking them with a 1, by derivative
        tics = np.concatenate( ( [0], np.where(mask ==1)[0]+1, [current.shape[0]] ) )
        del mask
        events = [ Segment(current=np.array(current), copy=True, 
                            start=tics[i],
                            duration=current.shape[0] ) for i, current in enumerate( np.split( current, tics[1:-1]) ) ]
        return [ event for event in self._lambda_select( events ) ]
    
    def GUI( self ):
        '''
        Override the default GUI for use in the Abada GUI, allowing for customization of the rules and threshol via
        the GUI. 
        '''
        threshDefault, timeDefault = "90", "1"
        maxCurrentDefault, minCurrentDefault = threshDefault, "-0.5" 

        grid = Qt.QGridLayout()
        
        threshLabel = Qt.QLabel( "Maximum Current" )
        threshLabel.setToolTip( "Raw ionic current threshold, which, if dropped below, indicates an event." ) 
        grid.addWidget( threshLabel, 0, 0 )

        self.threshInput = Qt.QLineEdit()
        self.threshInput.setText( threshDefault )
        grid.addWidget( self.threshInput, 0, 2, 1, 1 )

        minCurrentLabel = Qt.QLabel( "Minimum Current (pA):" )
        minCurrentLabel.setToolTip( "This sets a filter requiring all ionic current in an event be above this amount." )
        grid.addWidget( minCurrentLabel, 1, 0 )

        self.minCurrentInput = Qt.QLineEdit()
        self.minCurrentInput.setText( minCurrentDefault )
        grid.addWidget( self.minCurrentInput, 1, 2, 1, 1 )

        timeLabel = Qt.QLabel( "Time:" )
        timeLabel.setToolTip( "This sets a filter requiring all events are of a certain length." )
        grid.addWidget( timeLabel, 3, 0 ) 

        self.timeDirectionInput = Qt.QComboBox()
        self.timeDirectionInput.addItem( ">" )
        self.timeDirectionInput.addItem( "<" )
        grid.addWidget( self.timeDirectionInput, 3, 1 )

        self.timeInput = Qt.QLineEdit()
        self.timeInput.setText( timeDefault )
        grid.addWidget( self.timeInput, 3, 2, 1, 1 )
        return grid

    def set_params( self ):
        '''
        Read in the data from the GUI and use it to customize the rules or threshold of the parser. 
        '''
        self.rules = []
        self.threshold = float( self.threshInput.text() )
        self.rules.append( lambda event: event.max < self.threshold )
        if self.minCurrentInput.text() != '':
            self.rules.append( lambda event: event.min > float( self.minCurrentInput.text() ) )
        if self.timeInput.text() != '':
            if str( self.timeDirectionInput.currentText() ) == '<':
                self.rules.append( lambda event: event.duration < float( self.timeInput.text() ) )
            elif str( self.timeDirectionInput.currentText() ) == '>':
                self.rules.append( lambda event: event.duration > float( self.timeInput.text() ) )
        if self.rules == []:
            self.rules = None

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class SpeedyStatSplit( Parser ):
    '''
    See cparsers.pyx FastStatSplit for full documentation. This is just a
    wrapper for the cyton implementation to add a GUI.
    '''
    uiTemplate=[("sampling_freq", "spin", {'value': 200000, 'step': 1, 'dec':True, 'bounds': [0,None], 'suffix': 'Hz', 'siPrefix': True}),
             ("min_width", "spin", {"value": 100,'step': 1, 'dec':True, 'min':2, "int":True}),
             ("max_width", "spin", {"value":1000000,'step': 1, 'dec':True, "int":True}),
             ("window_width", "spin", {"value":10000, 'step': 1, 'dec':True, "int":True}),
            #  ("min_gain_per_sample", "(float,None,"hidden"),
            #  "false_positive_rate":(float,None,"hidden"),
            #  "prior_segments_per_second":(float,None,"hidden"),
             ("cutoff_freq", "spin", {"hidden":True})]
    _parse_return_type=(list,Segment)
    _parse_meta_return_type=(list,MetaSegment)
    
    _parse_inputs=[("Current",np.ndarray)]
    _parse_outputs=[("Segments",list)]
    
    
    def __init__( self, sampling_freq, min_width=100, max_width=1000000, window_width=10000, 
        min_gain_per_sample=None, false_positive_rate=None,
        prior_segments_per_second=None, cutoff_freq=None ):

        self.min_width = min_width
        self.max_width = max_width
        self.min_gain_per_sample = min_gain_per_sample
        self.window_width = window_width
        self.prior_segments_per_second = prior_segments_per_second
        self.false_positive_rate = false_positive_rate
        self.sampling_freq = sampling_freq
        self.cutoff_freq = cutoff_freq

    def parse( self, current ):
        parser = FastStatSplit( self.min_width, self.max_width, 
            self.window_width, self.min_gain_per_sample, self.false_positive_rate,
            self.prior_segments_per_second, self.sampling_freq, self.cutoff_freq )
        return parser.parse( current )
    def parse_meta( self, current ):
        parser = FastStatSplit( self.min_width, self.max_width, 
            self.window_width, self.min_gain_per_sample, self.false_positive_rate,
            self.prior_segments_per_second, self.sampling_freq, self.cutoff_freq )
        return parser.parse_meta( current )

    def best_single_split( self, current ):
        parser = FastStatSplit( self.min_width, self.max_width, 
            self.window_width, self.min_gain_per_sample, self.false_positive_rate,
            self.prior_segments_per_second, self.sampling_freq )
        return parser.best_single_split( current )

    def GUI( self ):
        grid = Qt.QGridLayout()
        grid.addWidget( Qt.QLabel( "Minimum Width (samples): "), 0, 0, 1, 3)
        grid.addWidget( Qt.QLabel( "Maximum Width (samples): " ), 1, 0, 1, 3 )
        grid.addWidget( Qt.QLabel( "Window Width (samples): " ), 2, 0, 1, 3 )
        grid.addWidget( Qt.QLabel( "Minimum Gain / Sample: " ), 3, 0, 1, 3 )
        
        self.minWidth = Qt.QLineEdit()
        self.minWidth.setText('1000')
        self.maxWidth = Qt.QLineEdit()
        self.maxWidth.setText('1000000')
        self.windowWidth = Qt.QLineEdit('10000')
        self.windowWidth.setText('10000')
        self.minGain = Qt.QLineEdit()
        self.minGain.setText('0.05')

        grid.addWidget( self.minWidth, 0, 3 )
        grid.addWidget( self.maxWidth, 1, 3 )
        grid.addWidget( self.windowWidth, 2, 3 )
        grid.addWidget( self.minGain, 3, 3 )
        return grid

    def set_params( self ):
        try:
            self.min_width = int(self.minWidth.text())
            self.max_width = int(self.maxWidth.text())
            self.window_width = int(self.windowWidth.text())
            self.min_gain_per_sample = float(self.minGain.text())
        except:
            pass


#########################################
# STATE PARSERS 
#########################################

class snakebase_parser( Parser ):
    '''
    A simple parser based on dividing when the peak-to-peak amplitude of a wave exceeds a certain threshold.
    '''

    def __init__( self, threshold=1.5 ):
        self.threshold = threshold

    def parse( self, current ):
        # Take the derivative of the current first
        diff = np.abs( np.diff( current ) )
        # Find the places where the derivative is low
        tics = np.concatenate( ( [0], np.where( diff < 1e-3 )[0], [ diff.shape[0] ] ) )
        # For pieces between these tics, make each point the cumulative sum of that piece and put it together piecewise
        cumsum = np.concatenate( ( [ np.cumsum( diff[ tics[i] : tics[i+1] ] ) for i in range( tics.shape[0]-1 ) ] ) )
        # Find the edges where the cumulative sum passes a threshold
        split_points = np.where( np.abs( np.diff( np.where( cumsum > self.threshold, 1, 0 ) ) ) == 1 )[0] + 1
        # Return segments which do pass the threshold
        return [ Segment( current = current[ tics[i]: tics[i+1] ], start = tics[i] ) for i in range( 1, tics.shape[0] - 1, 2 ) ]

    def GUI( self ):
        threshDefault = "1.5"

        grid = Qt.QGridLayout()
        grid.setVerticalSpacing(0)
        grid.addWidget( Qt.QLabel( "Threshold" ), 0, 0 )
        self.threshInput = Qt.QLineEdit()
        self.threshInput.setToolTip("Peak to peak amplitude threshold, which if gone above, indicates a state transition.")
        self.threshInput.setText( threshDefault ) 

        grid.addWidget( self.threshInput, 0, 1 )
        grid.addWidget( self.mergerThreshInput, 1, 1 )
        return grid

    def set_params( self ):
        self.threshold = float( self.threshInput.text() )

class FilterDerivativeSegmenter( Parser ):
    '''
    This parser will segment an event using a filter-derivative method. It will
    first apply a bessel filter at a certain cutoff to the current, then it will
    take the derivative of that, and segment when the derivative passes a
    threshold.
    '''

    def __init__( self, low_threshold=1, high_threshold=2, cutoff_freq=1000.,
        sampling_freq=1.e5 ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq

    def parse( self, current ):
        '''
        Apply the filter-derivative method to filter the ionic current.
        '''

        # Filter the current using a first order Bessel filter twice, one in
        # both directions to preserve phase
        from scipy import signal
        nyquist = self.sampling_freq / 2.
        b, a = signal.bessel( 1, self.cutoff_freq / nyquist, btype='low', analog=0, output='ba' )
        filtered_current = signal.filtfilt( b, a, np.array( current ).copy() )

        # Take the derivative
        deriv = np.abs( np.diff( filtered_current ) )

        # Find the edges of the blocks which fulfill pass the lower threshold
        blocks = np.where( deriv > self.low_threshold, 1, 0 )
        block_edges = np.abs( np.diff( blocks ) )
        tics = np.where( block_edges == 1 )[0] + 1 

        # Split points are points in the each block which pass the high
        # threshold, with a maximum of one per block 
        split_points = [0] 

        for start, end in it.izip( tics[:-1:2], tics[1::2] ): # For all pairs of edges for a block..
            segment = deriv[ start:end ] # Save all derivatives in that block to a segment
            if np.argmax( segment ) > self.high_threshold: # If the maximum derivative in that block is above a threshold..
                split_points = np.concatenate( ( split_points, [ start, end ] ) ) # Save the edges of the segment 
                # Now you have the edges of all transitions saved, and so the states are the current between these transitions
        tics = np.concatenate( ( split_points, [ current.shape[0] ] ) )
        tics = list(map( int, tics ))
        return [ Segment( current=current[ tics[i]:tics[i+1] ], start=tics[i] ) 
                    for i in range( 0, len(tics)-1, 2 ) ]

    def GUI( self ):
        lowThreshDefault = "1e-2"
        highThreshDefault = "1e-1"

        grid = Qt.QGridLayout()
        grid.addWidget( Qt.QLabel( "Low-pass Threshold: " ), 0, 0 )
        grid.addWidget( Qt.QLabel( "High-pass Threshold: " ), 1, 0 )

        self.lowThreshInput = Qt.QLineEdit()
        self.lowThreshInput.setText( lowThreshDefault )
        self.lowThreshInput.setToolTip( "The lower threshold, of which one maximum is found." )
        self.highThreshInput = Qt.QLineEdit()
        self.highThreshInput.setText( highThreshDefault )
        self.highThreshInput.setToolTip( "The higher threshold, of which the maximum must be abov." )

        grid.addWidget( self.lowThreshInput, 0, 1 )
        grid.addWidget( self.highThreshInput, 1, 1 )
        return grid

    def set_params( self ):
        self.low_thresh = float( self.lowThreshInput.text() )
        self.high_thresh = float( self.highThreshInput.text() )