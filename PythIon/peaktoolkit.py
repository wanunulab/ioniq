# -*- coding: utf8 -*-
"""
Written By: Ali Fallahi
6/22/2021
"""


import sys
import numpy as np
from peaktoolkitwidget import *
from PyQt6 import QtWidgets

class peakToolkit(QtWidgets.QWidget):

    def __init__(self, master=None):
        QtWidgets.QWidget.__init__(self)
        self.uipeak = Ui_peakWidget()
        self.uipeak.setupUi(self)
        
        self.uipeak.peakCancelBtn.clicked.connect(self.close)
        
    def close(self):
        if self.parent != None:
            self.hide()
        else: 
            self.destroy()

    
    
if __name__ == "__main__":
    global myapp_peaktoolkit
    app_peaktoolkit = QtWidgets.QApplication(sys.argv)
    myapp_peaktoolkit = peakToolkit()
    myapp_peaktoolkit.show()
    sys.exit(app_peaktoolkit.exec_())