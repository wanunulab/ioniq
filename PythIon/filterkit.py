# -*- coding: utf8 -*-
"""
Written By: Ali Fallahi
6/22/2021
"""


import sys
import numpy as np
from filterkitwidget import *
from PyQt5 import QtCore, QtGui, QtWidgets

class FilterKit(QtWidgets.QWidget):

    def __init__(self, master=None):
        QtWidgets.QWidget.__init__(self)
        self.uifilt = Ui_FilterWindow()
        self.uifilt.setupUi(self)
        
        # self.uifilt.peakCancelBtn.clicked.connect(self.close)
        
    def close(self):
        if self.parent != None:
            self.hide()
        else: 
            self.destroy()

    
    
if __name__ == "__main__":
    global myapp_filtkit
    app_filtkit = QtWidgets.QApplication(sys.argv)
    myapp_filtkit = FilterKit()
    myapp_filtkit.show()
    sys.exit(app_filtkit.exec_())