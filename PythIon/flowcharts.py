import pyqtgraph as pg
from pyqtgraph import flowchart
from pyqtgraph.flowchart.library.common import CtrlNode

from PythIon.Parsers.parsers import SpeedyStatSplit 

from pyqtgraph.flowchart.NodeLibrary import NodeLibrary
import pyqtgraph.flowchart.library as fclib


class AnalysisAbstractNode(CtrlNode):
    nodeName=None
    uiTemplate = []
    AnalysisClass=None
    def __init__(self,name):
        if self.AnalysisClass is None:
            raise ValueError("No analysis class passed in.")
            
        else:
            try:
                self.identify_structure(self.AnalysisClass)
            except:
                raise Exception("Invalid analysis class:", self.AnalysisClass)
                
        self.input_terminals=self._inputs
        self.output_terminals=self._outputs
        terminals={}
        for input in self.input_terminals:
            terminals.update({input[0]:{"io":"in"}})
        for output in self.output_terminals:
            terminals.update({output[0]:{"io":"out"}})
        
        super().__init__(name,terminals=terminals)
        
            
        
    
    def identify_structure(self,AnalysisClass):
        # self._params=AnalysisClass.get_init_params()
        self.__class__.uiTemplate=getattr(AnalysisClass,'uiTemplate')
        self._inputs=AnalysisClass.get_process_inputs()
        self._outputs=AnalysisClass.get_process_outputs()
        self._has_meta_process=hasattr(AnalysisClass,"parse_meta")
        if self._has_meta_process:
            self.uiTemplate.append( ("meta_output","check",{"checked":True}) )
            
        self.nodeName=AnalysisClass.get_name()
        
        
        
    def _recursive_dataset_traversal(self,data):
        if isinstance(data,list):
            return [self._recursive_dataset_traversal(self,item) for item in data]
        if isinstance(data,dict):
            if "name" in data.keys():
                if "segments" in data.keys():
                    return {"name":data["name"], "segments":[self.process_appropriate(segment) for segment in data["segments"]]}
                if "events" in data.keys():
                    return {}
    def processData(self,data):
        
        return None
    
    def process(self, In, display=True):
        analyzer=self.AnalysisClass(**dict([(ctrlitem[0],self.ctrls[ctrlitem[0]].value) for ctrlitem in self.__class__.uiTemplate]))
        results=analyzer.parse(In)
        return dict(zip([o[0] for o in self._outputs],results))
        
def get_lib():
    # library=NodeLibrary()
    library=fclib.LIBRARY.copy()
    
    SSLCtrlNode=type("NodeSpeedyStatSplit",(AnalysisAbstractNode,),{"AnalysisClass":SpeedyStatSplit,"nodeName":"SpeedyStatSplit"})
    library.addNodeType(SSLCtrlNode,paths=[("Parsers",)])
    return library
                    
    
        
        
        # self.output_type=AnalysisClass.get_output_type()
        
        
    

class PythionFlowchart(flowchart.Flowchart):
    def __init__(self, terminals=None, name=None, filePath=None, library=None):
        super().__init__(terminals, name, filePath, library)
        # self.widget().
        pass
    