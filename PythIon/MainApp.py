# import PyQt6
from pyqtgraph.Qt import QtCore, QtGui,QtWidgets
import pyqtgraph as pg
from pyqtgraph import dockarea
import sys,os
import json
import time
import numpy as np
from pyqtgraph import flowchart
from flowcharts import get_lib
import theme

# if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
#     PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    
# if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
#     # PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
#     # os.environ["QT_FONT_DPI"] = "96"
#     os.environ["QT_SCALE_FACTOR"] = "1.25"
class MainAppGUIBase(QtWidgets.QMainWindow):
    def __init__(self, master=None,docks_state=None):
        super().__init__(master)
        self.area=dockarea.DockArea()
        self.setCentralWidget(self.area)
        self.showMaximized()
        self.setup_docks(docks_state)

    def setup_docks(self,docks_state=None):
        if docks_state is None:
            self.setup_default_docks()
        else:
            self.area.restoreState(docks_state)
            
    def setup_default_docks(self):
        
        self.dock_app_controls=dockarea.Dock("App Controls",size=(1000,200),hideTitle=True,autoOrientation=False)
        self.dock_trace=dockarea.Dock("Traces",size=(1000,500))
        self.dock_pipeline_view=dockarea.Dock("Pipeline Flowchart",size=(1000,500))
        self.dock_nice_graphs=dockarea.Dock("Nice Graphs",size=(1000,500))
        self.dock_event_view=dockarea.Dock("Event View",size=(500,500))
        self.dock_quick_graphs=dockarea.Dock("Quick Graphs",size=(500,500))
        
        self.dock_pipeline_controls=dockarea.Dock("Pipeline Controls",size=(200,1200))
        
        self.area.addDock(self.dock_pipeline_controls,'right')
        self.area.addDock(self.dock_nice_graphs,'left')
        self.area.addDock(self.dock_pipeline_view,'above',self.dock_nice_graphs)
        self.area.addDock(self.dock_trace,'above',self.dock_pipeline_view)
        self.area.addDock(self.dock_app_controls,'top',self.dock_trace)
        self.area.addDock(self.dock_event_view,'bottom',self.dock_trace)
        self.area.addDock(self.dock_quick_graphs,'right',self.dock_event_view)
        
        
    def get_docks_state(self):
        return self.area.saveState()
    
    
    def setup_action_bar(self):
        pass

        

class MainAppGUI(MainAppGUIBase):
    def __init__(self, master=None, docks_state=None):
        super().__init__(master, docks_state)
        
        self.pipeline_controls=flowchart.Flowchart(terminals={
            'dataIn': {'io': 'in'},'dataOut': {'io': 'out'}    
            },library=get_lib())
        self.pipeline_controls.createNode("SpeedyStatSplit")
        self.dock_pipeline_controls.addWidget(self.pipeline_controls.widget())
        self.dock_pipeline_view.addWidget(self.pipeline_controls.widget().chartWidget)
        self.place_widgets()
        
        # self.dock_pipeline_controls
    def place_widgets(self):
        self.loadDataBtn=QtWidgets.QPushButton("Load Data")
        self.dock_app_controls.addWidget(self.loadDataBtn)
        self.loadDataBtn.clicked.connect(self.get_file)
        
        self.plot_trace=pg.PlotWidget(background='k')
        self.plot_trace.setLabel("left","Current", units="A",siPrefix=True)
        self.plot_trace.setLabel("bottom","Time", units="s",siPrefix=True)
        self.dock_trace.addWidget(self.plot_trace)
        self.plot_trace_node=self.pipeline_controls.createNode("PlotWidget")
        self.plot_trace_node.setPlot(self.plot_trace)
        
        
        
    def get_file(self):
        try:
            ######## attempt to open dialog from most recent directory########
            self.datafilename = QtWidgets.QFileDialog.getOpenFileName(self,'Open file',"",("*.log;*.opt;*.npy;*.abf;*.edh"))
            if self.datafilename != ('', ''):
                self.datafilename = self.datafilename[0]
                self.direc=os.path.dirname(self.datafilename)
                self.load_data()
        except IOError:
            #### if user cancels during file selection, exit loop#############
            pass
    def load_data(self):
        if str(os.path.splitext(self.datafilename)[1])=='.edh':
            self.headerfilename=self.datafilename

            basefname=str(os.path.splitext(self.datafilename)[0])
            i=0
            # max_limit = input("enter maximum number of files to import")
            datafilenames=[]
            while (os.path.exists(basefname+f"_{i:03}.dat")):
                datafilenames.append(basefname+f"_{i:03}.dat")
                i+=1
            i=0
            while (os.path.exists(basefname+f"_CH001_{i:03}.abf")):
                datafilenames.append(basefname+f"_CH001_{i:03}.abf")
                i+=1

            nfiles=len(datafilenames)
            start_num, ok = PyQt5.QtWidgets.QInputDialog.getInt(None,"Starting File",f"Enter starting file number to import (0 - {nfiles-1:03})",value=0,min=0, max=nfiles-1)
            if not ok:
                return
            max_limit, ok = PyQt5.QtWidgets.QInputDialog.getInt(None,"File limit","Enter maximum number of files to import",value=1)
            if not ok:
                return
            self.datafilenames=[]
            i=start_num
            while (os.path.exists(basefname+f"_{i:03}.dat") and i<start_num+max_limit):
                self.datafilenames.append(basefname+f"_{i:03}.dat")
                i+=1
            i=start_num
            while (os.path.exists(basefname+f"_CH001_{i:03}.abf") and i<start_num+max_limit):
                self.datafilenames.append(basefname+f"_CH001_{i:03}.abf")
                i+=1
            if self.datafilenames:
                for fname in self.datafilenames:
                    print("\t"+fname)
                    
            with open(self.headerfilename,"r") as headerfile:
                for line in headerfile:
                    if line.startswith("EDH Version"):
                        if line.split(":")[1].strip()!="2.0":
                            print("EDH version not supported")
                            return
                    else:
                        if line.startswith("Channels"):
                            self.numberOfChannels=int(line.split(":")[1])
                        if line.startswith("Sampling frequency"):
                            samplerate=np.float64(int(line.split(":")[1].strip().split()[0])*1000)
                        if line.startswith("Final Bandwidth"):
                            filtrate=np.float64(samplerate/int(line.split("/")[1].split(" ")[0]))
                        if line.startswith("Active Channels"):
                            self.numberOfChannels=int(line.split(":")[1])
            if self.datafilenames:
                self.data=[]
                self.voltage=[]
                self.matfilename=self.datafilenames[0]
                for datafilename in self.datafilenames:
                    if datafilename[-4:]=='.abf':
                        # data=pyabf.ABF(datafilename)
                        # data=data.data
                        return
                    else:
                        data=np.fromfile(datafilename,dtype="float32")
                        data=data.reshape((self.numberOfChannels+1,-1),order="F")
                    self.data=np.concatenate((self.data,data[0]*1e-9),axis=None)
                    self.voltage=np.concatenate((self.voltage,data[self.numberOfChannels]),axis=None)
                    self.is_normalized_to_G=False
                    print('voltage channel shape ',self.voltage.shape)
                    print('current channel shape ',self.data.shape)
            self.outputsamplerate=samplerate
            # if samplerate < self.outputsamplerate:
                # self.outputsamplerate=samplerate
                # self.ui.outputsamplerateentry.setText(str((round(samplerate)/1000)))
                
            self.pipeline_controls.setInput(dataIn=self.data)

        
        
        
        
        
def start():
    global myapp
    app = QtWidgets.QApplication(sys.argv)
    # resolution = app.desktop().screenGeometry()
    # width,height = resolution.width(), resolution.height()
    myapp = MainAppGUI()
    program_dir=os.getcwd()
    with open(os.path.join(program_dir,"PythIon/themes/maintheme.qss"), 'r') as qss:
        compiled_stylesheet=theme.apply_theme(qss.read(),theme.get_available_themes()[0])
        myapp.setStyleSheet(compiled_stylesheet)
    # time.sleep(0.1)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    start()