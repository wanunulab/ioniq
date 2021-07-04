# -*- coding: utf8 -*-
"""
Written By: Ali Fallahi
6/22/2021
"""


import sys
import numpy as np
from filterkitwidget import *
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets

from scipy import signal
from scipy import ndimage
from scipy import fft
import matplotlib.pyplot as plt

class FilterKit(QtWidgets.QWidget):

    def __init__(self, samplerate, master=None):
        QtWidgets.QWidget.__init__(self)
        self.outputsamplerate=samplerate
        self.uifilt = Ui_FilterWindow()
        self.uifilt.setupUi(self)
        self.uifilt.filterPreviewBtn.clicked.connect(self.preview)
        self.previewPlot=None
        self.filterReady=False
       
        
    def close(self):
        if self.parent != None:
            self.hide()
        else: 
            self.destroy()

    def generate_filter(self, ftype:str, fcutoff, forder:int,sigma=None):
        """
        Generate filter coefficients b,a (or window for gaussian)

        Parameters
        ----------
        ftype: str
            type of filter to use
            'besselLP' for bessel lowpass,
            'besselBandstop' for bessel band-stop
            'gaussian' for gaussian filter
        fcutoff : 
            single low-pass cutoff frequency :float if ftype=='besselLP'
            tuple of (low_cutoff:float, high_cutoff:float) if ftype=='besselBandstop'
            values should be given in hertz (Hz)

            ignored if ftype=='gaussian'
        forder : int
            order of the filter.
            must be set zero if ftype=='gaussian' otherwise the nth derivative of a gaussian window will be returned
        sigma: float
            sigma (standard deviation) of gaussian window.
            value should be given in seconds
            ignored if ftype != "gaussian"
        Returns
        -------
        b,a : 
            Numerator (b) and denominator (a) polynomials of the IIR bessel filter.

        window: numpy array of size int(8.0*sigma*samplerate+1)
            for example a sigma of 40e-6 s with sample rate of 200kHz produces a gaussian kernel of shape (65,)
        """
        if ftype=="besselLP":
            lo_end_over_Nyquist = fcutoff/(self.outputsamplerate/2)#*2*np.pi

            # b,a = signal.iirfilter(forder,Wn=[lo_end_over_Nyquist],btype="lowpass",ftype="bessel",output="ba",analog=False)
            b,a = signal.bessel(forder,Wn=[lo_end_over_Nyquist],btype="lowpass",output="ba",analog=False,norm='mag')
            
            self.Wn=[b,a]
            self.filterReady=True
            return b,a
        elif ftype=="besselBandstop":
            lo_end_over_Nyquist = fcutoff[0]/(self.outputsamplerate/2)#*2*np.pi
            hi_end_over_Nyquist = fcutoff[1]/(self.outputsamplerate/2)#*2*np.pi
            b,a = signal.iirfilter(forder,Wn=[lo_end_over_Nyquist,hi_end_over_Nyquist],btype="bandstop",ftype="bessel",output="ba",analog=False)
            self.Wn=[b,a]
            self.filterReady=True
            return b,a
        elif ftype=="gaussian":
            window=ndimage.filters._gaussian_kernel1d(sigma*self.outputsamplerate,order=forder,radius=int(4.0*sigma*self.outputsamplerate+0.5))
            self.Wn=window
            self.filterReady=True
            return window

    def process_filter_params(self):
        self.sigma=None
        self.fcutoff=None
        if self.uifilt.filterLowpassBessel.isChecked():
            self.ftype="besselLP"
            self.forder=self.uifilt.besselLowpassOrder.value()
            self.fcutoff=self.uifilt.besselCutoffFreq.value()*1000
        elif self.uifilt.filterBandstopBessel.isChecked():
            self.ftype="besselBandstop"
            self.forder=self.uifilt.besselBandstopOrder.value()
            self.fcutoff=(self.uifilt.besselBandstopLow.value()*1000,self.uifilt.besselBandstopHigh.value()*1000)
        elif self.uifilt.filterGaussian.isChecked():
            self.ftype="gaussian"
            self.forder=0
            self.sigma=self.uifilt.filterGaussianSigma.value()*1e-6


        
        self.Wn=self.generate_filter(self.ftype,self.fcutoff,self.forder,self.sigma)
        return

    def preview(self):
        try:
            self.process_filter_params()
            if not self.filterReady:
                print("could not process filter")
                return
            print(self.Wn)
            if self.previewPlot is None or self.previewPlot:
                self.previewPlot = plt.figure("Filter Response")

            if self.ftype!="gaussian":
                # w,h=signal.freqs(self.Wn[0],self.Wn[1],worN=np.logspace(2, np.log10(2*np.pi*self.outputsamplerate/2), 4096*8))
                # w=w/(2*np.pi)
                w,h=signal.freqz(self.Wn[0],self.Wn[1],fs=self.outputsamplerate,worN=np.logspace(2, np.log10(self.outputsamplerate/2), 4096*8))
                wgd,group_delay=signal.group_delay(self.Wn,fs=self.outputsamplerate,w=w)
                if self.ftype=="besselLP":
                    filterlabel=f"Ord.{self.forder} Bess Lowpass $f_c = {self.fcutoff/1000} kHz$"
                elif self.ftype=="besselBandstop":
                    filterlabel=f"Ord.{self.forder} Bess Bandstop $f_{{lo}} = {self.fcutoff[0]/1000:.2f} kHz, f_{{hi}} = {self.fcutoff[1]/1000:.2f} kHz$"
            
            elif self.ftype=="gaussian":
                h=fft.rfft(self.Wn,4096*8)/(self.Wn.shape[0]/2)
                w=np.arange(0,h.shape[0])/h.shape[0]*self.outputsamplerate/2
                
                group_delay=-np.diff(np.unwrap(np.angle(h))/(2*np.pi))/np.diff(w)
                h=h/np.max(np.abs(h))
                
                filterlabel="$\sigma = "+f"{self.sigma*1e6} us$"
                wgd=w[1:]
            if True:
                ax1=plt.subplot(311)
                plt.plot(w,np.abs(h),label=filterlabel)
                plt.legend(fontsize='small')
                plt.semilogx()
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude")
                plt.grid(b=True,which='both', axis='both')
                print(w,h)
                plt.draw()
                
                
                ax2=plt.subplot(312,sharex=ax1)
                plt.plot(w,np.unwrap(np.angle(h))*360/(2*np.pi),label=filterlabel)
                #w[1:], -np.diff(np.unwrap(np.angle(h)))/np.diff(w)
                #plt.legend()
                plt.semilogx()
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("unwrapped phase (deg)")
                plt.grid(b=True,which='both', axis='both')

                plt.draw()

                ax3=plt.subplot(313,sharex=ax1)


                plt.plot(wgd,group_delay,label=filterlabel)
                # w[1:], -np.diff(np.unwrap(np.angle(h)))/np.diff(w)
                #plt.legend()
                plt.semilogx()
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("group delay")
                plt.grid(b=True,which='both', axis='both')
                self.previewPlot.show()



        except Exception as e:
            print(e)

    def apply_to(self,data):
        try:
            print("apply to called")
            self.process_filter_params()
            if not self.filterReady:
                return None
            if self.ftype!="gaussian":
                print(data,self.Wn)
                return signal.filtfilt(self.Wn[0],self.Wn[1],data,axis=0)
            else:
                return ndimage.correlate1d(data, self.Wn[::-1], axis=-1, output=None, mode="reflect")

        except Exception as e:
            print(e)
            return None


            


    
    
if __name__ == "__main__":
    global myapp_filtkit,sample_fig

    def apply_to_sample():
        print("applying to sample")
        rng = np.random.default_rng(seed=1)
        t=np.linspace(0,1,200001)
        data=rng.random(t.shape)+10

        filteredData=myapp_filtkit.apply_to(data)
        print("applied")
        if filteredData is not None:
            # print(filteredData.shape)
            # print(filteredData,data)
            plt.figure("Sample Filtering",clear=True)
            plt.plot(t[::10],data[::10],'k',label="Original Signal",lw=1)
            plt.plot(t[::10],filteredData[::10],'b',label="Filtered Signal")
            plt.legend()
            plt.draw()
            plt.show()

            plt.figure("power spectrum",clear=True)
            f,pxx=signal.welch(data,fs=200e3,scaling='spectrum',nperseg=1024)
            plt.semilogx(f,pxx,label="raw signal")
            f,pxx=signal.welch(filteredData,fs=200e3,scaling='spectrum',nperseg=1024)
            plt.semilogx(f,pxx,label="filtered signal")
            plt.legend(fontsize='small')
            plt.draw()
            plt.show()





    
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app_filtkit = QtWidgets.QApplication(sys.argv)
    myapp_filtkit = FilterKit(samplerate=200e3)
    myapp_filtkit.show()
    myapp_filtkit.uifilt.filterApplyBtn.clicked.connect(apply_to_sample)


    sys.exit(app_filtkit.exec_())