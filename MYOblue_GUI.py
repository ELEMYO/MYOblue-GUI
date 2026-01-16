# Graphical interface for signal visualization and interaction with ELEMYO MYOblue sensors
# 2025-10-19 by ELEMYO https://github.com/ELEMYO/MYOblue-GUI
# 
# Code is placed under the MIT license
# Copyright (c) 2021 ELEMYO
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ===============================================

import sys
import os
from importlib import metadata
import subprocess

print(">>> MYOblue_GUI is launching. Please wait...")

required = {'pyserial', 'pyqtgraph', 'PyQt5', 'numpy', 'scipy'}

installed = {dist.metadata['Name'].lower() for dist in metadata.distributions()}

missing = {pkg for pkg in required if pkg.lower() not in installed}

if missing:
    print(">>> Installing missing libraries:", missing)

    for module in list(missing):
        print(f">>> Installing {module}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", module],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            missing.remove(module)
        else:
            print(f">>> \"{module}\" NOT installed successfully.")
            print(">>> Please check your internet connection or contact support: info@elemyo.com")

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QEvent
import serial
import pyqtgraph as pg
import numpy as np
import time
from scipy.signal import butter, lfilter
import serial.tools.list_ports
from scipy.fftpack import fft
from serial import SerialException
from datetime import datetime
import struct
from configparser import ConfigParser

# Main window
class GUI(QtWidgets.QMainWindow):
    # Initialize constructor
    def __init__(self):
          super(GUI, self).__init__()
          self.initUI()
    # Custom constructor 
    def initUI(self):        
        self.setWindowTitle("ELEMYO MYOblue GUI v1.2.1")
        self.setWindowIcon(QtGui.QIcon('img/icon.png'))
        self.delay = 0.25 # Graphics update delay
        self.NUM_SENSORS = 8 
        self.cfg = ConfigParser()
        self.cfg.optionxform = str
        self.cfg.read("config.ini")
        self.fs = self.cfg.getint("APPLICATION", "SampleRate_(HZ)")  # Sampling frequency in Hz
        self.dt = [1/self.fs]*self.NUM_SENSORS  # Time between two signal measurements in s
        for i in range(self.NUM_SENSORS): self.dt[i] = self.cfg.getfloat(f"SENSOR{i+1}", "dt_(s)")
        
        self.timeWidth = 10 # Plot window length in seconds
        self.dataWidth = int((self.timeWidth + 2)*self.fs) # Maximum count of plotting data points
        self.data = Data(self.NUM_SENSORS, self.dataWidth)
        self.l = [0]*self.NUM_SENSORS # Current sensor data point
        self.FFT = np.zeros((self.NUM_SENSORS, 500)) # Fast Fourier transform data
        
        self.MovingAverage = MovingAverage(self.fs) 
        self.bandstop_filter_50Hz = bandstop_filter_50Hz(self.fs)
        self.bandstop_filter_60Hz = bandstop_filter_60Hz(self.fs)
        self.bandpass_filter = bandpass_filter(1, self.fs/2-1, self.fs)
        self.HP_filter = HP_filter(1, self.fs)
        
        self.recordingFileName_BIN = '' # Recording file name
        self.recordingFileName_TXT = '' # Recording file name
        self.recordingFile_BIN = 0 # Recording file 
        self.recordingFile_TXT = 0 # Recording file
        self.loadFileName = '' # Data load file name
        self.loadFile = 0 # Data load variable
        self.sliderpos = 0 # Position of data slider 
        self.loadDataLen = 0 # Number of signal samples in data file
        self.loadData = 0 # Data from load file
        
        # Accessory variables for data read from serial
        self.TIMER = 0;
        self.ms_len = [0]*self.NUM_SENSORS;
        self.msg_end = bytearray([0])
        
        self.VDD = [0]*self.NUM_SENSORS # Battery charge array (in voltes) 
        self.MSG_NUM = [0]*self.NUM_SENSORS
        self.MSG_NUM_0 = [0]*self.NUM_SENSORS
        
        # Accessory variables for EMG mask
        self.FlagEMG = [0]*self.NUM_SENSORS
        self.num = [0]*self.NUM_SENSORS
        self.Fl = 1
        
        self.mean = [1]*self.NUM_SENSORS
        self.meanN = [0]*self.NUM_SENSORS

        # Menu panel
        self.liveFromSerialAction = QtWidgets.QAction(QtGui.QIcon('img/play.png'), 'Start/Stop live from serial ', self)
        self.liveFromSerialAction.setCheckable(True)
        self.liveFromSerialAction.setChecked(False)
        self.liveFromSerialAction.triggered.connect(self.liveFromSerial)
        
        self.COMports=QtWidgets.QComboBox()
        self.COMports.setDisabled(False)
        
        self.refreshAction = QtWidgets.QAction(QtGui.QIcon('img/refresh.png'), 'Refresh screen (R)', self)
        self.refreshAction.setShortcut('r')
        self.refreshAction.triggered.connect(self.refreshForAction)
        self.refreshAction.setDisabled(True)   
        
        self.dataRecordingAction = QtWidgets.QAction(QtGui.QIcon('img/rec.png'), 'Start/Stop recording', self)
        self.dataRecordingAction.triggered.connect(self.dataRecording)
        self.dataRecordingAction.setCheckable(True)
        self.dataRecordingAction.setDisabled(True)
        
        self.pauseAction = QtWidgets.QAction(QtGui.QIcon('img/pause.png'), 'Pause (Space)', self)
        self.pauseAction.setCheckable(True)
        self.pauseAction.setChecked(False)
        self.pauseAction.triggered.connect(self.pause)
        self.pauseAction.setShortcut('Space')
               
        self.PlaybackAction = QtWidgets.QAction(QtGui.QIcon('img/playback.png'), 'Start/Stop playback from file: \nFILE NOT SELECTED', self)
        self.PlaybackAction.triggered.connect(self.Playback)
        self.PlaybackAction.setCheckable(True)
        self.PlaybackAction.setDisabled(True)
        
        dataLoadAction = QtWidgets.QAction(QtGui.QIcon('img/load.png'), 'Select playback file', self)
        dataLoadAction.triggered.connect(self.dataLoad)

        self.passLowFreq = QtWidgets.QSpinBox()
        self.passLowFreq.setRange(2, int(self.fs/2) -10)
        self.passLowFreq.setValue(self.cfg.getint("APPLICATION", "BandPassFilterLF"))
        self.passLowFreq.setDisabled(True)
                      
        self.passHighFreq = QtWidgets.QSpinBox()
        self.passHighFreq.setRange(10, int(self.fs/2) -1)
        self.passHighFreq.setValue(self.cfg.getint("APPLICATION", "BandPassFilterHF"))
        self.passHighFreq.setDisabled(True)     
        
        self.slider = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.slider.setValue(0)
        self.slider.setFixedWidth(40)
        self.slider.setDisabled(True)

        self.sensorsNumberAction = QtWidgets.QLabel(' SENSORS NUMBER: ', self)
        self.sensorsNumberAction1 = QtWidgets.QLabel('     ', self)
        self.sensorsNumber = QtWidgets.QDoubleSpinBox()
        self.sensorsNumber.setRange(1, self.NUM_SENSORS)
        self.sensorsNumber.setDecimals(0)
        self.sensorsNumber.setDisabled(True)
        self.sensorsNumber.setValue(self.cfg.getint("APPLICATION", "SensorsNumber")) 
        
        self.rawSignalAction = QtWidgets.QCheckBox('RAW EMG', self)
        self.rawSignalAction.setChecked(self.cfg.getboolean("APPLICATION", "RAW_EMG"))
        self.rawSignalAction.toggled.connect(self.rawSignalActionTriggered)
        self.rawSignalAction1 = QtWidgets.QLabel('       ', self)       
        
        self.rectificationSignalAction = QtWidgets.QCheckBox('RECTIFICATION', self)
        self.rectificationSignalAction.setChecked(self.cfg.getboolean("APPLICATION", "Rectification"))
        self.rectificationSignalAction.toggled.connect(self.rectificationSignalActionTriggered)
        self.rectificationSignalAction1 = QtWidgets.QLabel('     ', self)     
        
        self.EnvelopeSignalAction = QtWidgets.QCheckBox('ENVELOPE:', self)
        self.EnvelopeSignalAction.setChecked(self.cfg.getboolean("APPLICATION", "Envelope"))
        self.EnvelopeSignalAction1 = QtWidgets.QLabel('    ', self)
        self.EnvelopeSignalAction2 = QtWidgets.QLabel('      ', self)
        self.envelopeSmoothingCoefficient = QtWidgets.QDoubleSpinBox()
        self.envelopeSmoothingCoefficient.setSingleStep(0.01)
        self.envelopeSmoothingCoefficient.setRange(0, 1)
        self.envelopeSmoothingCoefficient.setValue(self.cfg.getfloat("APPLICATION", "EnvelopeSmoothingCoefficient"))
        
        self.RMSsignalAction = QtWidgets.QCheckBox('RMS:', self)
        self.RMSsignalAction.setChecked(self.cfg.getboolean("APPLICATION", "RMS"))
        self.RMSsignalAction1 = QtWidgets.QLabel('    ', self)
        self.RMSsignalAction2 = QtWidgets.QLabel('      ', self)
        self.RMSinterval = QtWidgets.QDoubleSpinBox()
        self.RMSinterval.setSingleStep(0.01)
        self.RMSinterval.setRange(0.01, 3)
        self.RMSinterval.setValue(self.cfg.getfloat("APPLICATION", "RMSinterval"))
                
        self.bandstopAction = QtWidgets.QCheckBox('BANDSTOP FILTER:', self)
        self.bandstopAction.setChecked(self.cfg.getboolean("APPLICATION", "BandStopFilter"))
        
        self.notchActiontypeBox=QtWidgets.QComboBox()
        self.notchActiontypeBox.addItem("50 Hz")
        self.notchActiontypeBox.addItem("60 Hz")
        self.notchActiontypeBox.setDisabled(True)
                        
        self.bandpassAction = QtWidgets.QCheckBox('BANDPASS FILTER:', self)
        self.bandpassAction.setChecked(self.cfg.getboolean("APPLICATION", "BandPassFilter"))
        self.bandpassAction1 = QtWidgets.QLabel('  -  ', self)
        self.bandpassAction2 = QtWidgets.QLabel('       ', self)
        

        self.sensorSelectedAction = QtWidgets.QLabel('Sensor: ', self)
        self.sensorSelectedAction.setStyleSheet("background-color: transparent; font-weight: bold;")

        self.sensorSelectedActionBox=QtWidgets.QComboBox()
        self.sensorSelectedActionBox.addItem("1")
        self.sensorSelectedActionBox.setStyleSheet("background-color: gray; font-weight: bold;")

#--------------------------        
        # Toolbar
        toolbar = []
        toolbar.append(self.addToolBar('Tool1'))
        toolbar.append(self.addToolBar('Tool2'))
        toolbar.append(self.addToolBar('Tool3'))
        
        
        widgets = [self.COMports, self.liveFromSerialAction, self.dataRecordingAction, self.refreshAction, self.pauseAction]
        for w in widgets:
            if isinstance(w, QtWidgets.QAction): toolbar[0].addAction(w)
            elif isinstance(w, QtWidgets.QWidget): toolbar[0].addWidget(w)
            
        widgets = [dataLoadAction, self.PlaybackAction, self.slider]
        for w in widgets:
            if isinstance(w, QtWidgets.QAction): toolbar[1].addAction(w)
            elif isinstance(w, QtWidgets.QWidget): toolbar[1].addWidget(w)
        
        widgets = [self.sensorsNumberAction, self.sensorsNumber, self.rawSignalAction1, self.rawSignalAction, self.rectificationSignalAction1, self.rectificationSignalAction,
                   self.EnvelopeSignalAction1, self.EnvelopeSignalAction, self.envelopeSmoothingCoefficient, self.EnvelopeSignalAction2,
                   self.RMSsignalAction1, self.RMSsignalAction, self.RMSinterval, self.RMSsignalAction2,
                   self.bandstopAction, self.notchActiontypeBox, self.bandpassAction2, self.bandpassAction, self.passLowFreq,
                   self.bandpassAction1, self.passHighFreq]
        for w in widgets:
            if isinstance(w, QtWidgets.QAction): toolbar[2].addAction(w)
            elif isinstance(w, QtWidgets.QWidget): toolbar[2].addWidget(w)
       
        self.pw = []
        for i in range(self.NUM_SENSORS):
            self.pw.append(CustomPlotWidget())
            self.pw[i].addItem(self.pw[i].coord_label, ignoreBounds=True)  
        self.pw[7].getAxis('bottom').setStyle(showValues=True)        
        
        # Plot widget for spectral Plot
        self.pwFFT = pg.PlotWidget(background=(13, 13, 13, 255))
        self.pwFFT.showGrid(x=True, y=True, alpha=0.7) 
        self.pFFT = self.pwFFT.plot()
        self.pFFT.setPen(color=(100, 255, 255), width=1)
        self.pwFFT.setLabel('bottom', 'Frequency', 'Hz')
                
        # Histogram widget
        self.pb = [] # Histogram item array, index - sensor number
        self.pbar = pg.PlotWidget(background=(13 , 13, 13, 255))
        self.pbar.showGrid(x=True, y=True, alpha=0.7)  
        colors = [(153, 0, 0), (229, 104, 19), (221, 180, 10), (30, 180, 30), (11, 50, 51), (29, 160, 191), (30, 30, 188), (75, 13, 98)]   
        
        # Numbering of graphs
        backLabel = []
        numberLabel = []
        
        for i in range(self.NUM_SENSORS):  
            self.pb.append(pg.BarGraphItem(x=np.linspace(i+1, i+2, num=1), height=np.linspace(i+1, i+2, num=1), width=0.3, pen=colors[i], brush=colors[i]))
            self.pbar.addItem(self.pb[i])  
            numberLabel.append(QtWidgets.QLabel(" " + str(i+1) + " "))
            r, g, b = colors[i]
            numberLabel[i].setStyleSheet(f"font-size: 25px; background-color: rgb({r}, {g}, {b}); border-radius: 14px;")
            backLabel.append(QtWidgets.QLabel(""))
            backLabel[i].setStyleSheet("font-size: 25px; background-color: rgb(21, 21, 21);")
        self.pbar.setLabel('bottom', 'Sensor number')
        
        # Style
        centralStyle = "color: rgb(255, 255, 255); background-color: rgb(13, 13, 13);"
        
        self.ChargeLabel  = []
        for i in range(self.NUM_SENSORS):
            self.ChargeLabel.append(QtWidgets.QLabel("BATTERY: 0.00 V"))
            self.ChargeLabel[i].setStyleSheet("background-color: transparent; font-weight: bold;")
        
        self.TriggerLabel  = []
        self.TriggerValue  = []
        self.NumberEMG_Lable = []
        
        self.StartTimeValue  = []
        self.NumberEMG = []
        
        for i in range(self.NUM_SENSORS):
            self.TriggerLabel.append(QtWidgets.QLabel("Trigger value:"))
            self.NumberEMG_Lable.append(QtWidgets.QLabel("Number of contr.:"))
            
            self.TriggerLabel[i].setStyleSheet("background-color: transparent; font-weight: bold; color: rgba(255, 255, 255, 0.5);")
            self.NumberEMG_Lable[i].setStyleSheet("background-color: transparent; font-weight: bold; color: rgba(255, 255, 255, 0.5);")
            
            self.TriggerValue.append(QtWidgets.QSpinBox())
            self.TriggerValue[i].setSingleStep(1)
            self.TriggerValue[i].setRange(0, 2500)
            self.TriggerValue[i].setValue(self.cfg.getint(f"SENSOR{i+1}", "Trigger_value"))
            
            self.NumberEMG.append(QtWidgets.QSpinBox())
            self.NumberEMG[i].setSingleStep(1)
            self.NumberEMG[i].setRange(0, 10000)
            self.NumberEMG[i].setValue(0)

        # Main widget
        centralWidget = QtWidgets.QWidget()
        centralWidget.setStyleSheet(centralStyle)
        
        self.textWindow = QtWidgets.QPlainTextEdit()
        self.textWindow.setReadOnly(True)
        
        self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "program launched\n")
        
        # Layout
        vbox = QtWidgets.QVBoxLayout()
        
        topleft = QtWidgets.QFrame()
        topleft.setFrameShape(QtWidgets.QFrame.StyledPanel)
        
        plotLayout = []
        self.row = []
        for i in range(self.NUM_SENSORS):
            plotLayout.append(QtWidgets.QGridLayout())
            plotLayout[i] = QtWidgets.QGridLayout()
            if i % 2 == 0: plotLayout[i].addWidget(backLabel[int(i/2)], 0, 0, 10, 1)
            plotLayout[i].addWidget(numberLabel[i], 0, 0, 10, 1, Qt.AlignVCenter)
            plotLayout[i].addWidget(self.pw[i], 0, 1, 10, 50)
            plotLayout[i].addWidget(self.ChargeLabel[i], 0, 49) 
            plotLayout[i].addWidget(self.TriggerLabel[i], 1, 49) 
            plotLayout[i].addWidget(self.TriggerValue[i], 1, 50)   
            plotLayout[i].addWidget(self.NumberEMG_Lable[i], 2, 49) 
            plotLayout[i].addWidget(self.NumberEMG[i], 2, 50) 
            plotLayout[i].setContentsMargins(0, 0, 0, 0)    
            
            self.row.append(QtWidgets.QWidget())
            self.row[i].setLayout(plotLayout[i])
            
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        splitter.handle(100)
        for i in range(self.NUM_SENSORS): splitter.addWidget(self.row[i])
        
        layout = QtWidgets.QGridLayout()       
        layout.addWidget(splitter, 0, 0, 40, 4)
        layout.addWidget(self.pbar, 0, 4, 20, 11)
        layout.addWidget(self.pwFFT, 20, 4, 16, 11)
        layout.setColumnStretch(2, 2)

        layout.addWidget(self.sensorSelectedAction , 20, 13, 1, 1)
        layout.addWidget(self.sensorSelectedActionBox , 20, 14, 1, 1)  
        
        layout.addWidget(self.textWindow, 37, 4, 3, 12)  
        
        layout.addWidget(self.textWindow, 36, 4, 3, 12)   
        
        vbox.addLayout(layout)
        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)  
        self.showMaximized()
        self.show()    
        
        # Serial monitor
        self.serialMonitor = SerialMonitor(self.delay)
        ports = [self.COMports.itemText(i) for i in range(self.COMports.count())]
        
        for i in range(len(self.serialMonitor.ports)):
                if self.serialMonitor.ports[i] not in ports:
                    self.COMports.addItem(self.serialMonitor.ports[i])
                    
        if self.serialMonitor.COM != '':
            self.serialMonitor.serialConnect()
            self.liveFromSerialAction.setChecked(True)
            self.dataRecordingAction.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live from " + self.serialMonitor.COM +" \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.COMports.setDisabled(True)
            self.refreshAction.setDisabled(False)

        self.sensorsNumber.valueChanged.connect(self.setSensorsNumber)
        self.setSensorsNumber(self.cfg.getint("APPLICATION", "SensorsNumber"))
        self.mainrun = MainRun(self.delay)
        self.mainrun.bufferUpdated.connect(self.updateListening, QtCore.Qt.QueuedConnection)  
        print(">>> MYOblue_GUI was launched successfully.")        
    
    def rawSignalActionTriggered(self):
        if self.rawSignalAction.isChecked():
            self.rectificationSignalAction.setCheckState(False)
    
    def rectificationSignalActionTriggered(self):
        if self.rectificationSignalAction.isChecked():
            self.rawSignalAction.setCheckState(False)
    
    def liveFromSerial(self):
        if self.liveFromSerialAction.isChecked():
            self.refresh()
            self.serialMonitor.serialConnect()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live from " + self.serialMonitor.COM +" \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.PlaybackAction.setChecked(False)
            self.refreshAction.setDisabled(False)   
            self.pauseAction.setDisabled(False)
            self.dataRecordingAction.setDisabled(False)
            self.COMports.setDisabled(True)
            self.slider.setDisabled(True)
            self.slider.setFixedWidth(40)
            self.sensorsNumber.setDisabled(False)
        else:
            self.refresh()
            self.serialMonitor.serialDisconnection()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live stopped\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.refreshAction.setDisabled(True)   
            self.pauseAction.setDisabled(True)
            self.dataRecordingAction.setDisabled(True)
            self.COMports.setDisabled(False)
            self.sensorsNumber.setDisabled(True)
           
    # Start working
    def start(self):
        self.mainrun.running = True
        self.mainrun.start()
    
    # Pause data plotting
    def pause(self):
        if self.pauseAction.isChecked():
            self.mainrun.running = False
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "pause ON" + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
        else:
            self.mainrun.running = True
            self.mainrun.start()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "pause OFF" + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)

    # Refresh data
    def refresh(self):
        self.l = [0] * self.NUM_SENSORS
        self.dataWidth = int((self.timeWidth + 2)*self.fs)
        self.data.refresh(self.dataWidth)
        self.msg_end = bytearray([0])      
        self.MSG_NUM = [0]*self.NUM_SENSORS
        self.ms_len =  [0]*self.NUM_SENSORS
        self.MSG_NUM_0 = [0]*self.NUM_SENSORS
        self.slider.setValue(0)
        self.sliderpos = 0
        self.TIMER = 0
        self.FFT = np.zeros((self.NUM_SENSORS, 500)) 
        for i in range(self.NUM_SENSORS):
            self.NumberEMG[i].setValue(0)
            self.FlagEMG[i] = 0
            self.num[i] = 0

    # Refresh screen
    def refreshForAction(self):
        self.refresh()
        self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "refresh" + "\n")
        self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)      
         
    # Initialize recording data to a file
    def dataRecording(self):
        if (self.dataRecordingAction.isChecked()):
            self.sensorsNumber.setDisabled(True)
            self.refreshAction.setDisabled(True)  
            self.recordingFileName_TXT = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
            self.recordingFileName_BIN = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".bin"
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "recording to \"" + os.getcwd() +"\\" + self.recordingFileName_BIN + "\"\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.recordingFile_TXT = open(self.recordingFileName_TXT, "a") # Data file creation
            self.recordingFile_TXT.write(datetime.now().strftime("Date: %Y.%m.%d\rTime: %H:%M:%S") + "\r\n") # Data file name
            self.recordingFile_TXT.write("File format: \r\n8 sensors data in mkV\r\n") # Data file format
            self.recordingFile_BIN = open(self.recordingFileName_BIN, 'ab')
        else:
            if not self.PlaybackAction.isChecked():
                self.refreshAction.setDisabled(False)
            self.recordingFile_TXT.close()
            self.recordingFile_BIN.close()
            self.sensorsNumber.setDisabled(False)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "recording stopped. Result file: \"" + os.getcwd() + self.recordingFileName_TXT + "\"\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
                
    # Selecting playback file
    def dataLoad(self):
        if self.liveFromSerialAction.isChecked():
            self.dataRecordingAction.setChecked(False)
            self.refreshAction.setDisabled(False)    
            self.pauseAction.setDisabled(False)
        self.recordingFileName_TXT = ''
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a file', '',
                                        'All Files (*.bin*)')
        if path != ('', ''):
            self.loadFileName = str(path[0])
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback file selected: " + self.loadFileName + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.PlaybackAction.setText("Start/Stop playback from file: \n" + self.loadFileName)
            self.PlaybackAction.setDisabled(False)
    
    # Playback initialization 
    def Playback(self):
        if self.PlaybackAction.isChecked():
            self.dataRecordingAction.setChecked(False)
            self.slider.setDisabled(False)
            self.slider.setFixedWidth(300)
            if self.liveFromSerialAction.isChecked():
                self.liveFromSerialAction.setChecked(False)
            self.refresh()
            self.liveFromSerialAction.setChecked(False)
            self.serialMonitor.serialDisconnection()
            self.dataRecordingAction.setDisabled(False)  
            self.refreshAction.setDisabled(True) 
            self.pauseAction.setDisabled(False)  
            self.COMports.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            self.loadFile = open(self.loadFileName, 'rb')
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback from: " + self.loadFileName + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.loadData = self.loadFile.read()
            self.loadDataLen = int(len(self.loadData)/16)
            self.loadFile.close()
            
        else:
            self.slider.setDisabled(True)
            self.slider.setFixedWidth(40)
            self.refresh()
            self.dataRecordingAction.setDisabled(True)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback stopped \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.pauseAction.setDisabled(True)  

    # Update
    def updateListening(self):
        self.cfg.set("APPLICATION", "RAW_EMG", str(self.rawSignalAction.isChecked()))
        self.cfg.set("APPLICATION", "Rectification", str(self.rectificationSignalAction.isChecked()))
        self.cfg.set("APPLICATION", "Envelope", str(self.EnvelopeSignalAction.isChecked()))
        self.cfg.set('APPLICATION', 'EnvelopeSmoothingCoefficient', str(self.envelopeSmoothingCoefficient.value()))
        self.cfg.set("APPLICATION", "RMS", str(self.RMSsignalAction.isChecked()))
        self.cfg.set('APPLICATION', 'RMSinterval', str(self.RMSinterval.value()))
        self.cfg.set("APPLICATION", "BandStopFilter", str(self.bandstopAction.isChecked()))
        self.cfg.set("APPLICATION", "BandPassFilter", str(self.bandpassAction.isChecked()))
        self.cfg.set("APPLICATION", "BandPassFilterLF", str(self.passLowFreq.value()))
        self.cfg.set("APPLICATION", "BandPassFilterHF", str(self.passHighFreq.value()))
        self.cfg.set("APPLICATION", "SampleRate_(HZ)", str(self.fs))
        for i in range (self.NUM_SENSORS): 
            self.cfg.set(f"SENSOR{i+1}", "dt_(s)", str(self.dt[i]))
            self.cfg.set(f"SENSOR{i+1}", "Trigger_value", str(self.TriggerValue[i].value()))
        
        if (not self.liveFromSerialAction.isChecked()):
            self.serialMonitor.updatePorts()
                   
            ports = [self.COMports.itemText(i) for i in range(self.COMports.count())]
            
            for i in range(self.COMports.count()):
                if self.COMports.itemText(i) not in self.serialMonitor.ports:
                    self.COMports.removeItem(i)
                    
            for i in range(len(self.serialMonitor.ports)):
                if self.serialMonitor.ports[i] not in ports:
                    self.COMports.addItem(self.serialMonitor.ports[i])
            
            if self.serialMonitor.COM != self.COMports.currentText():
                self.serialMonitor.COM = self.COMports.currentText()
                self.serialMonitor.connect = False
        
        if self.passLowFreq.value() > self.passHighFreq.value(): self.passLowFreq.setValue(self.passHighFreq.value())
        
        if self.bandpassAction.isChecked():
            self.passLowFreq.setDisabled(False)
            self.passHighFreq.setDisabled(False)
        else:
            self.passLowFreq.setDisabled(True)
            self.passHighFreq.setDisabled(True)
        
        if self.bandstopAction.isChecked(): 
            self.notchActiontypeBox.setDisabled(False)
        else:
            self.notchActiontypeBox.setDisabled(True)
            
        if self.EnvelopeSignalAction.isChecked():
            self.envelopeSmoothingCoefficient.setDisabled(False)
            self.MovingAverage.MA_alpha = self.envelopeSmoothingCoefficient.value()
        else:
            self.envelopeSmoothingCoefficient.setDisabled(True)
            
        if self.RMSsignalAction.isChecked():
            self.RMSinterval.setDisabled(False)
        else:
            self.RMSinterval.setDisabled(True)
        
        # Read data from File               
        if (self.PlaybackAction.isChecked() and self.loadFileName != ''):
            self.readFromFile()
        
        # Read data from serial          
        if (self.liveFromSerialAction.isChecked()):
            self.readFromSerial()
            for i in range (self.NUM_SENSORS):
                if self.ms_len[i]/119!=0:
                    self.mean[i] += self.ms_len[i]/119
                    self.meanN[i] += 1
                    
                    if (self.mean[i]/self.meanN[i]) < 2 and self.fs > 900:
                        self.fs = 500
                        self.dt = [1/self.fs]*self.NUM_SENSORS
                        self.dataWidth = int((self.timeWidth + 2)*self.fs) # Maximum count of plotting data points
                        self.passLowFreq.setRange(2, int(self.fs/2) -1)
                        self.passHighFreq.setRange(2, int(self.fs/2) -1)
                        self.bandpass_filter = bandpass_filter(1, self.fs/2-1, self.fs)
                        self.refresh()
                        return
                    
                    if (self.mean[i]/self.meanN[i]) > 2 and self.fs < 900:
                        self.fs = 1000
                        self.dt = [1/self.fs]*self.NUM_SENSORS
                        self.dataWidth = int((self.timeWidth + 2)*self.fs)
                        self.passLowFreq.setRange(2, int(self.fs/2) -1)
                        self.passHighFreq.setRange(2, int(self.fs/2) -1)
                        self.bandpass_filter = bandpass_filter(1, self.fs/2-1, self.fs)
                        self.refresh()
                        return

        while self.sensorSelectedActionBox.count() < int(self.sensorsNumber.value()): 
            self.sensorSelectedActionBox.addItem(str(self.sensorSelectedActionBox.count() + 1))
            
        while self.sensorSelectedActionBox.count() > int(self.sensorsNumber.value()): 
            self.sensorSelectedActionBox.removeItem(self.sensorSelectedActionBox.count()-1)

        if (self.PlaybackAction.isChecked() and self.loadFileName != '') or (self.liveFromSerialAction.isChecked()):  
            max_time = 0
            for i in range( int(self.sensorsNumber.value()) ):
                self.data.plot[i] = np.roll(self.data.raw[i], -self.l[i])
                self.data.plot[i] = ((self.data.plot[i] - 8192)/16384.0*2.49)*2000
                self.data.timePlot[i] = np.roll(self.data.time[i], -self.l[i])
            
                if self.bandstopAction.isChecked():
                    if (self.notchActiontypeBox.currentText() == "50 Hz"): self.data.plot[i] = self.bandstop_filter_50Hz.applay(self.data.plot[i], 1/self.dt[i])
                    if (self.notchActiontypeBox.currentText() == "60 Hz"): self.data.plot[i] = self.bandstop_filter_60Hz.applay(self.data.plot[i], 1/self.dt[i])
                                
                if (self.bandpassAction.isChecked()): 
                    self.data.plot[i] = self.bandpass_filter.applay(self.data.plot[i], self.passLowFreq.value(), self.passHighFreq.value(), 1/self.dt[i])
                    self.data.rectification[i] = abs(self.data.plot[i])
                else: self.data.rectification[i] = abs(self.HP_filter.applay(self.data.plot[i], 1, 1/self.dt[i]))
                
                if self.bandstopAction.isChecked() or self.bandpassAction.isChecked(): self.data.plot[i][0:int(1.5*self.fs)] = 0
                self.data.rectification[i][0:int(1.5*self.fs)] = 0
                
                # Shift the boundaries of the graph
                if (self.data.time[i][self.l[i] - 1] > max_time): max_time = self.data.time[i][self.l[i] - 1]
                self.pw[i].setXRange(self.timeWidth*((max_time)// self.timeWidth), self.timeWidth*((max_time) // self.timeWidth + 1))
                
                # Plot raw data or rectification
                if  self.rawSignalAction.isChecked(): self.pw[i].p.setData(y=self.data.plot[i], x=self.data.timePlot[i])
                elif  self.rectificationSignalAction.isChecked(): self.pw[i].p.setData(y=self.data.rectification[i], x=self.data.timePlot[i])
                elif not self.rawSignalAction.isChecked(): self.pw[i].p.clear()
                
                # Plot envelope data
                if  self.EnvelopeSignalAction.isChecked(): self.pw[i].pe.setData(y=self.data.envelope[i], x=self.data.timePlot[i])
                else: self.pw[i].pe.clear()     
                
                # Plot RMS data
                if  self.RMSsignalAction.isChecked(): self.pw[i].pi.setData(y=self.data.RMS[i], x=self.data.timePlot[i])
                else: self.pw[i].pi.clear()
                
                # Plot histogram
                self.pb[i].setOpts(height=2*self.data.RMS[i][-1])
                
                if self.ms_len[i] > 0:
                    self.data.envelope[i] = np.roll(self.data.envelope[i], - self.ms_len[i])
                    self.data.RMS[i] = np.roll(self.data.RMS[i], - self.ms_len[i])
                    n = int(self.RMSinterval.value()*1000/2)
                    
                    for j in range (self.dataWidth - self.ms_len[i], self.dataWidth):
                        self.data.envelope[i][j] = self.MovingAverage.movingAverage(i, self.data.rectification[i][j])
                        
                        if j >= n + 1:
                            I1 = (self.data.envelope[i][j-n]**2 + self.data.envelope[i][j-n-1]**2)*self.dt[i]*0.5
                            I2 = (self.data.envelope[i][j]**2 + self.data.envelope[i][j-1]**2)*self.dt[i]*0.5
                            self.data.RMS[i][j] = abs((self.data.RMS[i][j-1]**2 + (I2 - I1)/self.RMSinterval.value()))**0.5
                        else:
                            self.data.RMS[i][j] = 0
                            
                        if (self.FlagEMG[i] == 0) & (self.data.RMS[i][j] >= self.TriggerValue[i].value()):
                            self.FlagEMG[i] = 1
                            self.NumberEMG[i].setValue(self.NumberEMG[i].value() + 1)
                        
                        if (self.FlagEMG[i] == 1) & (self.data.RMS[i][j] < self.TriggerValue[i].value()):
                            self.FlagEMG[i] = 0

            for i in range( int(self.sensorsNumber.value()), self.NUM_SENSORS):
                self.pw[i].p.clear()
                self.pw[i].pe.clear()
                self.pw[i].pi.clear()
                self.pb[i].setOpts(height=0)
            
            # Plot FFT data
            Y = [0]*500
            i = int(self.sensorSelectedActionBox.currentIndex())
            Y = abs(fft(self.data.plot[i][-501: -1]))/500
            self.FFT[i] = (1-0.5)*Y + 0.5*self.FFT[i]
            X = 1/self.dt[i]*np.linspace(0, 1, 500)
            sensor = self.sensorSelectedActionBox.currentIndex()
            self.pFFT.setData(y=self.FFT[sensor][2: int(len(self.FFT[sensor])/2)], x=X[2: int(len(X)/2)])

            if (self.dataRecordingAction.isChecked()):
                DataRec = np.zeros((self.NUM_SENSORS, max(self.ms_len)))
                DataRecBin = np.zeros((self.NUM_SENSORS, max(self.ms_len)))
                flag = 0
                
                Data = np.zeros((self.NUM_SENSORS, self.dataWidth))
                
                for i in range( int(self.sensorsNumber.value()) ): Data[i] = np.roll(self.data.raw[i], -self.l[i])
                
                for i in range(self.NUM_SENSORS): 
                    if self.num[i] == 0: self.num[i] = self.dataWidth
                    if self.num[i] > self.ms_len[i]: self.num[i] -= self.ms_len[i]                 
            
                maxTime = np.max(list(map(max, self.data.timePlot)))
                for i in range(self.NUM_SENSORS):
                    if self.num[i] < 0 and self.ms_len[i] > 0: self.num[i] = max(self.num)
                    if (maxTime > self.data.timePlot[i][self.dataWidth - 1] + 2): self.num[i] = -1
                        
                if (max(self.num) > 0.8*self.dataWidth): self.Fl = 0
                if (max(self.num) < 0.6*self.dataWidth):  self.Fl = 1
                
                for i in range(self.NUM_SENSORS):                        
                    if (self.num[i] >= 0 ) and (self.num[i] <= self.dataWidth - max(self.ms_len)) and self.Fl == 1:
                        DataRec[i] = self.data.plot[i][self.num[i]: self.num[i] + max(self.ms_len)]   
                        DataRecBin[i] = Data[i][self.num[i]: self.num[i] + max(self.ms_len)] 
                        self.num[i] += max(self.ms_len) 
                        flag = 1
                        
                if flag == 1:
                    for i in range(max(self.ms_len)):
                        sensors_data = str(round(DataRec[0][i]))
                        for j in range(1, self.NUM_SENSORS): sensors_data += (" " + str(round(DataRec[j][i])))
                        sensors_data += '\n'
                        self.recordingFile_TXT.write(sensors_data)
                        
                        bin_data = struct.pack("H H H H H H H H", int(DataRecBin[0][i]), int(DataRecBin[1][i]), int(DataRecBin[2][i]), int(DataRecBin[3][i]), 
                                               int(DataRecBin[4][i]), int(DataRecBin[5][i]), int(DataRecBin[6][i]), int(DataRecBin[7][i]))             
                        self.recordingFile_BIN.write(bin_data)            

        else:
            for i in range(int(self.sensorsNumber.value())):
                self.pw[i].p.clear()
                self.pw[i].pe.clear()
                self.pw[i].pi.clear()
                self.pb[i].setOpts(height=0)
            self.pFFT.clear()   

    # Read data from File   
    def readFromFile(self): 
        self.ms_len = [0]*self.NUM_SENSORS
        
        j = 0
        while j < 200:
            j += 1
            
            if ( self.sliderpos > self.loadDataLen - 2):
                self.refresh()
                self.sliderpos = 0
                self.slider.setValue(0) 
                        
            unpeck_b = struct.unpack("H H H H H H H H", self.loadData[self.sliderpos*16:(self.sliderpos+1)*16])
            for i in range(self.NUM_SENSORS): 
                if ( self.l[i] == self.dataWidth):
                    self.l[i] = 0
                self.data.raw[i][self.l[i]] = unpeck_b[i]
                self.data.time[i][self.l[i]] = self.data.time[i][self.l[i]-1] + 1/self.fs
                self.l[i] = self.l[i] + 1
                if (self.ms_len[i] < self.dataWidth): self.ms_len[i] += 1 
            
            if ((self.slider.value() != int(self.sliderpos/self.loadDataLen*100))):
                self.sliderpos += int(self.slider.value()*self.loadDataLen/100 - self.sliderpos)
                temp = self.l
                temp_sliderpos = self.sliderpos
                self.refresh()
                self.l = temp
                self.sliderpos = temp_sliderpos
                for i in range(self.NUM_SENSORS): self.data.time[i][self.l[i]-1] = self.sliderpos*(1/self.fs)
                     
            self.sliderpos += 1
            self.slider.setValue(int(self.sliderpos/self.loadDataLen*100))

    # Read data from serial                  
    def readFromSerial(self): 
        self.ms_len = [0]*self.NUM_SENSORS
        
        msg = self.serialMonitor.serialRead() 
        TIME = time.perf_counter()
        
        # Parsing data from serial buffer
        if (len(msg) > 7):
            if (len(self.msg_end) > 1):
                msg =  self.msg_end + msg
                self.msg_end = bytearray([0])
            
            if (len(msg) % (246) != 0):
                if(len(msg)>250):
                    for i in range(len(msg) - 250, len(msg)-1, 1):
                       if (msg[i] == 0xFF) and (msg[i+1] == 0xFF):
                           self.msg_end = msg[i:]
                           msg = msg[0:i]
                           break
            
            if (len(msg) % 246 == 0):
                for msg_i in range(0, len(msg), 246):
                    sensorNum = int(msg[msg_i+2])-1 
                    if (sensorNum > 7): break
                    MSG_NUM = int(msg[msg_i+3] | msg[msg_i+4] << 8 | msg[msg_i+5] << 16)
                    
                    if self.MSG_NUM_0[sensorNum] == 0: 
                        self.MSG_NUM_0[sensorNum] = MSG_NUM
                        if self.TIMER == 0:
                            self.TIMER = time.perf_counter()
                        self.data.time[sensorNum][self.l[sensorNum]-1] = TIME - self.TIMER
                            
                    if MSG_NUM - self.MSG_NUM_0[sensorNum] > 0:
                        self.MSG_NUM[sensorNum] += MSG_NUM - self.MSG_NUM_0[sensorNum]
                        self.data.time[sensorNum][self.l[sensorNum]-1] += self.dt[sensorNum]*119*(MSG_NUM - self.MSG_NUM_0[sensorNum] - 1)
                        self.MSG_NUM_0[sensorNum] = MSG_NUM
                        MSG_NUM = self.MSG_NUM[sensorNum]
                    else:
                        self.MSG_NUM_0[sensorNum] = MSG_NUM
                        MSG_NUM = max(self.MSG_NUM)
                        self.MSG_NUM[sensorNum] = MSG_NUM
                        self.data.time[sensorNum][self.l[sensorNum]-1] = TIME - self.TIMER

                    self.VDD[sensorNum] = round(int(msg[msg_i+6] | msg[msg_i+7] << 8)/16384*0.6*6*2, 2)
                    string = "BATTERY: " + str(self.VDD[int(msg[msg_i+2])-1]) + " V"
                    while len(string) < 13:
                        string += "0"
                    self.ChargeLabel[int(msg[msg_i+2])-1].setText(string)
                   
                    if (self.VDD[int(msg[msg_i+2])-1]) > 2.5:
                        self.ChargeLabel[int(msg[msg_i+2])-1].setStyleSheet("color: green; background-color: transparent; font-weight: bold;")
                    else:
                        self.ChargeLabel[int(msg[msg_i+2])-1].setStyleSheet("color: red; background-color: transparent; font-weight: bold;")
                    
                    if TIME > self.TIMER:
                        for i in range(msg_i+8, msg_i + 246, 2):
                            if ( self.l[sensorNum] == self.dataWidth):
                                self.l[sensorNum] = 0 
                                if (self.dataRecordingAction.isChecked()):
                                    self.recordingFile_BIN.close()
                                    self.recordingFile_TXT.close()
                                    self.recordingFile_BIN = open(self.recordingFileName_BIN, 'ab')
                                    self.recordingFile_TXT = open(self.recordingFileName_TXT, "a")
                                    
                            self.data.raw[sensorNum][self.l[sensorNum]]  = int(msg[i] | msg[i+1] << 8)
                            if ( self.l[sensorNum] > 0):
                                self.data.time[sensorNum][self.l[sensorNum]] = self.data.time[sensorNum][self.l[sensorNum] - 1] + self.dt[sensorNum] 
                            else:
                                self.data.time[sensorNum][self.l[sensorNum]] = self.data.time[sensorNum][self.dataWidth - 1] + self.dt[sensorNum]

                            self.l[sensorNum] += 1
                            if (self.ms_len[sensorNum] < self.dataWidth): self.ms_len[sensorNum] += 1 

                        timeDifference = (TIME - self.TIMER) - self.data.time[sensorNum][self.l[sensorNum]-1]
                        if timeDifference > 0.2 and timeDifference < 0.4:
                            if timeDifference > 0.3: self.data.time[sensorNum][:] += 0.2
                            else:    
                                self.data.time[sensorNum][:] += 0.005
                                if (self.data.time[sensorNum][self.l[sensorNum]-1] > 10): self.dt[sensorNum]*=1.001
                        if timeDifference < -0.2 and timeDifference > -1:
                            if timeDifference < -0.3: self.data.time[sensorNum][:] -= 0.2
                            else:
                                self.data.time[sensorNum][:] -= 0.005
                                if (self.data.time[sensorNum][self.l[sensorNum]-1] > 10): self.dt[sensorNum]*=0.999

    def setSensorsNumber(self, num):
        
        self.cfg.set('APPLICATION', 'SensorsNumber', str(int(num)))
        with open('config.ini', 'w') as f:
            self.cfg.write(f)
        
        if self.liveFromSerialAction.isChecked():
            self.refresh()
        
        for i in range(self.NUM_SENSORS):
            self.row[i].hide()
            self.pw[i].getAxis('bottom').setStyle(showValues=False)
            self.pw[i].showLabel('bottom', 0)
            self.pw[i].getAxis('bottom').setStyle(showValues=False)
        
        self.pw[int(num)-1].getAxis('bottom').setStyle(showValues=True)
        
        self.pbar.clear()
        for i in range(int(num)):  
            self.pbar.addItem(self.pb[i])  
            self.row[i].show()
   
    # Exit event
    def closeEvent(self, event):
        with open('config.ini', 'w') as f:
            self.cfg.write(f)
        self.mainrun.running = False
        self.serialMonitor.serialDisconnection()
        event.accept()     

class Data:
    def __init__(self, NUM_SENSORS, dataWidth):
        self.dataWidth = dataWidth
        self.NUM_SENSORS = NUM_SENSORS
        self.raw = np.zeros((self.NUM_SENSORS, self.dataWidth)) 
        self.plot = np.zeros((self.NUM_SENSORS, self.dataWidth))
        self.envelope = np.zeros((self.NUM_SENSORS, self.dataWidth))
        self.RMS = np.zeros((self.NUM_SENSORS, self.dataWidth))
        self.rectification = np.zeros((self.NUM_SENSORS, self.dataWidth))
        self.time = np.zeros((self.NUM_SENSORS, self.dataWidth)) 
        self.timePlot = np.zeros((self.NUM_SENSORS, self.dataWidth))
    def refresh(self, dataWidth):
        self.dataWidth = dataWidth
        self.raw = np.zeros((self.NUM_SENSORS, self.dataWidth)) 
        self.plot = np.zeros((self.NUM_SENSORS, self.dataWidth))
        self.envelope = np.zeros((self.NUM_SENSORS, self.dataWidth))
        self.RMS = np.zeros((self.NUM_SENSORS, self.dataWidth))
        self.rectification = np.zeros((self.NUM_SENSORS, self.dataWidth))
        self.time = np.zeros((self.NUM_SENSORS, self.dataWidth)) 
        self.timePlot = np.zeros((self.NUM_SENSORS, self.dataWidth))

# Butterworth bandpass filter
class bandpass_filter:
    def __init__ (self, lowcut, highcut, fs):
        self.order = 4
        self.fs = fs
        self.lowcut = lowcut/(0.5*fs)
        self.highcut = highcut/(0.5*fs)
        self.b, self.a = butter(self.order, [self.lowcut, self.highcut], btype='bandpass')
        
    def applay(self, data, lowcut, highcut, fs):
        if self.lowcut!=lowcut or self.highcut!=highcut or self.fs != fs:
            self.fs = fs
            self.lowcut = lowcut/(0.5*fs)
            self.highcut = highcut/(0.5*fs)
            self.b, self.a = butter(self.order, [self.lowcut, self.highcut], btype='bandpass')
        return lfilter(self.b, self.a, data)

# Butterworth bandstop filter
class bandstop_filter_50Hz:
    def __init__(self, fs):
        self.order = 4
        self.fs = fs
        self.a = [0]*4
        self.b = [0]*4
        for i in range(4):
            lowcut = (48+50*i)/(0.5*self.fs)
            highcut =(52+50*i)/(0.5*self.fs)
            self.b[i], self.a[i] = butter(self.order, [lowcut, highcut], btype='bandstop')
            
    def applay(self, data, fs):
        if self.fs != fs:
            self.fs = fs
            self.a = [0]*4
            self.b = [0]*4
            for i in range(4):
                lowcut = (48+50*i)/(0.5*self.fs)
                highcut =(52+50*i)/(0.5*self.fs)
                self.b[i], self.a[i] = butter(self.order, [lowcut, highcut], btype='bandstop')
        for i in range(4):
            data = lfilter(self.b[i], self.a[i], data)
        return data

# Butterworth bandstop filter
class bandstop_filter_60Hz:
    def __init__(self, fs):
        self.order = 4
        self.fs = fs
        self.a = [0]*4
        self.b = [0]*4
        for i in range(4):
            lowcut = (58+60*i)/(0.5*self.fs)
            highcut =(62+60*i)/(0.5*self.fs)
            self.b[i], self.a[i] = butter(self.order, [lowcut, highcut], btype='bandstop')
            
    def applay(self, data, fs):
        if self.fs != fs:
            self.fs = fs
            self.a = [0]*4
            self.b = [0]*4
            for i in range(4):
                lowcut = (58+60*i)/(0.5*self.fs)
                highcut =(62+60*i)/(0.5*self.fs)
                self.b[i], self.a[i] = butter(self.order, [lowcut, highcut], btype='bandstop')
        for i in range(4):
            data = lfilter(self.b[i], self.a[i], data)
        return data
# Butterworth bandpass filter
class HP_filter:
    def __init__ (self, lowcut, fs):
        self.order = 4
        self.fs = fs
        self.lowcut = lowcut/(0.5*fs)
        self.b, self.a = butter(self.order, self.lowcut, btype='highpass')
        
    def applay(self, data, lowcut, fs):
        if self.lowcut!=lowcut or self.fs != fs:
            self.fs = fs
            self.lowcut = lowcut/(0.5*fs)
            self.b, self.a = butter(self.order, self.lowcut, btype='highpass')
        return lfilter(self.b, self.a, data)

# Moving average class
class MovingAverage:
    # Custom constructor
    def __init__(self, fs):
        self.MA = np.zeros((8, 3)) 
        self.MA_alpha = 0.95
        self.fs = fs
    
    def movingAverage(self, i, data):
        self.MA[i][0] = (1 - self.MA_alpha)*data + self.MA_alpha*self.MA[i][0];
        self.MA[i][1] = (1 - self.MA_alpha)*(self.MA[i][0]) + self.MA_alpha*self.MA[i][1];
        self.MA[i][2] = (1 - self.MA_alpha)*(self.MA[i][1]) + self.MA_alpha*self.MA[i][2];
        return self.MA[i][2]*2

class CustomPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(background=(21 , 21, 21, 255))
        self.getAxis('left').setWidth(40)
        self.showGrid(x=True, y=True, alpha=0.7) 
        self.coord_label=pg.TextItem(anchor=(0, 1), color='w')
        
        self.p = self.plot()
        self.pe = self.plot()
        self.pi = self.plot()
        
        self.p.setPen(color=(100, 255, 255), width=0.8)
        self.pe.setPen(color=(255, 0, 0), width=1)
        self.pi.setPen(color=(0, 255, 0), width=1)
        self.getAxis('bottom').setStyle(showValues=False)
        
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self.onMouseMove)

    def leaveEvent(self, event: QEvent):
        
        self.coord_label.setVisible(False)
        super().leaveEvent(event)

    def enterEvent(self, event: QEvent):
        
        self.coord_label.setVisible(True)
        super().enterEvent(event)
        
    def onMouseMove(self, evt):
        pos = evt[0]
        if not self.sceneBoundingRect().contains(pos):
            self.coord_label.setText('')
            return
        
        vb = self.plotItem.vb
        mp = vb.mapSceneToView(pos)
        vr = vb.viewRect()
        self.coord_label.setHtml("<div style='background: rgba(21 , 21, 21, 255); color: gray; ""font-weight: bold; font-size: 11px; padding: 6px; ""border-radius: 4px;'>"f"X={mp.x():.3f}, Y={mp.y():.3f}</div>")       

        self.coord_label.setPos(vr.left(), vr.top())


# Serial monitor class
class SerialMonitor:
    # Custom constructor
    def __init__(self, delay):
        self.running = False
        self.connect = False
        self.baudRate = 1000000
        self.playFile = 0
        self.delay = delay      
        self.ports = [p[0] for p in serial.tools.list_ports.comports(include_links=False) ]
        self.COM = ''
        self.ser = serial.Serial()
        if len(self.ports) > 0:
            self.COM = self.ports[0]
        
    def updatePorts(self):
        self.ports = [p[0] for p in serial.tools.list_ports.comports(include_links=False) ]
    
    def serialConnect(self):
        self.updatePorts()
        if not self.connect:
            if self.COM != '':
                try:
                    self.ser = serial.Serial(self.COM, self.baudRate)
                    self.connect = True  
                except SerialException :
                    self.connect = False
                    
    def serialDisconnection(self):
        self.ser.close()
        self.connect = False
        
    def serialRead(self):  
        msg = bytes(0)
        try:
            msg = self.ser.read( self.ser.inWaiting() )
        except SerialException :
            try:
               self.ser.close()
               self.ser.open()
               msg = bytes(0)
            except SerialException :
                pass
            pass
        return msg

# Serial monitor class
class MainRun(QtCore.QThread):
    bufferUpdated = QtCore.pyqtSignal()
    # Custom constructor
    def __init__(self, delay):
        QtCore.QThread.__init__(self)
        self.running = False
        self.playFile = 0
        self.delay = delay      

    # Listening port
    def run(self):
        while self.running is True:
            self.bufferUpdated.emit()
            time.sleep(self.delay) 
         
# Starting program       
if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.show()
    window.start()
    sys.exit(app.exec_())

