# Graphical interface for signal visualization and interaction with ELEMYO MYOblue sensors
# 2026-08-06 by ELEMYO https://github.com/ELEMYO/MYOblue-GUI
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
from PyQt5.QtCore import Qt, pyqtSignal

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
from PyQt5.QtGui import QPen, QColor

# Main window
class GUI(QtWidgets.QMainWindow):
    # Initialize constructor
    def __init__(self):
          super(GUI, self).__init__()
          self.initUI()
    # Custom constructor 
    def initUI(self):   
        self.setWindowTitle("ELEMYO MYOblue GUI v1.2.2")
        self.BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.REC_DIR = os.path.join(self.BASE_DIR, "rec")
        os.makedirs(self.REC_DIR, exist_ok=True) 
        self.setWindowIcon(QtGui.QIcon(os.path.join(self.BASE_DIR, 'img', 'icon.png')))
        self.delay = 0.120 # Graphics update delay
        self.NUM_SENSORS = 8 
        self.cfg = ConfigParser()
        self.cfg.optionxform = str
        self.cfg.read(os.path.join(self.BASE_DIR, "config.ini"))
        self.fs = self.cfg.getint("APPLICATION", "SampleRate_(HZ)")  # Sampling frequency in Hz
        if not (990 <= self.fs <= 1010): self.fs = 1000
        self.dt = [1/self.fs]*self.NUM_SENSORS  # Time between two signal measurements in s
        for i in range(self.NUM_SENSORS): 
            self.dt[i] = self.cfg.getfloat(f"SENSOR{i+1}", "dt_(s)")
            if not (0.00099 <= self.dt[i] <= 0.00101): self.dt[i] = 0.001
        
        self.v_time = [0.0] * self.NUM_SENSORS       
        self.pll_initialized = [False] * self.NUM_SENSORS 
        self.sensor_uptime = [0.0] * self.NUM_SENSORS
        
        self.timeWidth = 10 # Plot window length in seconds
        self.dataWidth = int((self.timeWidth + 2)*self.fs) # Maximum count of plotting data points
        self.data = Data(self.NUM_SENSORS, self.dataWidth)
        self.l = [0]*self.NUM_SENSORS # Current sensor data point
        self.FFT = np.zeros((self.NUM_SENSORS, 500), dtype=np.float32) # Fast Fourier transform data
        
        self.MovingAverage = MovingAverage(self.fs) 
        self.bandstop_filter_50Hz = bandstop_filter_50Hz(self.fs)
        self.bandstop_filter_60Hz = bandstop_filter_60Hz(self.fs)
        self.bandpass_filter = bandpass_filter(1, self.fs/2-1, self.fs)
        self.HP_filter = HP_filter(1, self.fs)
        
        self.recordingFileName_BIN = '' # Recording file name
        self.recordingFileName_TXT = '' # Recording file name
        self.recordingFile_BIN = None # Recording file 
        self.recordingFile_TXT = None # Recording file
        self.loadFileName = '' # Data load file name
        self.loadFile = 0 # Data load variable
        self.sliderpos = 0 # Position of data slider 
        self.loadDataLen = 0 # Number of signal samples in data file
        self.loadData = 0 # Data from load file
        self.markers_list = []
        
        # Accessory variables for data read from serial
        self.TIMER = 0;
        self.TIMER_temp = 0;
        self.ms_len = [0]*self.NUM_SENSORS;
        self.msg_end = bytearray([0])
        
        self.VDD = [0]*self.NUM_SENSORS # Battery charge array (in voltes)
        self.MSG_NUM_0 = [0]*self.NUM_SENSORS
        
        # Accessory variables for EMG mask
        self.FlagEMG = [0]*self.NUM_SENSORS
        self.num = [0]*self.NUM_SENSORS
        self.Fl = 1
        
        self.mean = [1]*self.NUM_SENSORS
        self.meanN = [0]*self.NUM_SENSORS

        # Menu panel
        self.liveFromSerialAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(self.BASE_DIR, 'img', 'play.png')), 'Start/Stop live from serial ', self)
        self.liveFromSerialAction.setCheckable(True)
        self.liveFromSerialAction.setChecked(False)
        self.liveFromSerialAction.triggered.connect(self.liveFromSerial)
        
        self.COMports=QtWidgets.QComboBox()
        self.COMports.setDisabled(False)
        
        self.refreshAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(self.BASE_DIR, 'img', 'refresh.png')), 'Refresh screen (R)', self)
        self.refreshAction.setShortcut('r')
        self.refreshAction.triggered.connect(self.refreshForAction)
        self.refreshAction.setDisabled(True)   
        
        self.dataRecordingAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(self.BASE_DIR, 'img', 'rec.png')), 'Start/Stop recording', self)
        self.dataRecordingAction.triggered.connect(self.dataRecording)
        self.dataRecordingAction.setCheckable(True)
        self.dataRecordingAction.setDisabled(True)
        
        self.pauseAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(self.BASE_DIR, 'img', 'pause.png')), 'Pause (Space)', self)
        self.pauseAction.setCheckable(True)
        self.pauseAction.setChecked(False)
        self.pauseAction.triggered.connect(self.pause)
        self.pauseAction.setShortcut('Space')
               
        self.PlaybackAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(self.BASE_DIR, 'img', 'playback.png')), 'Start/Stop playback from file: \nFILE NOT SELECTED', self)
        self.PlaybackAction.triggered.connect(self.Playback)
        self.PlaybackAction.setCheckable(True)
        self.PlaybackAction.setDisabled(True)
        
        dataLoadAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(self.BASE_DIR, 'img', 'load.png')), 'Select playback file', self)
        dataLoadAction.triggered.connect(self.dataLoad)

        self.passLowFreq = ClampedSpinBox()
        self.passLowFreq.setRange(1, int(self.fs/2) -50)
        lf_value = self.cfg.getint("APPLICATION", "BandPassFilterLF")
        if not 1 <= lf_value <= 450: lf_value = 10
        self.passLowFreq.setValue(lf_value)
        self.passLowFreq.setDisabled(not self.cfg.getboolean("APPLICATION", "BandPassFilter"))
                              
        self.passHighFreq = ClampedSpinBox()
        self.passHighFreq.setRange(10, int(self.fs/2) -10)
        
        self.passHighFreq.valueChanged.connect( lambda val: self.passLowFreq.setValue(min(self.passLowFreq.value(), val)))
            
        hf_value = self.cfg.getint("APPLICATION", "BandPassFilterHF")
        if not lf_value <= hf_value <= int(self.fs/2) -10: hf_value = 490 
        self.passHighFreq.setValue(hf_value)
        self.passHighFreq.setDisabled(not self.cfg.getboolean("APPLICATION", "BandPassFilter"))  
        
        self.slider = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        self.slider.setValue(0)
        self.slider.setFixedWidth(40)
        self.slider.setDisabled(True)

        self.sensorsNumberAction = QtWidgets.QLabel(' SENSORS NUMBER: ', self)
        self.sensorsNumberAction1 = QtWidgets.QLabel('     ', self)
        self.sensorsNumber = QtWidgets.QDoubleSpinBox()
        self.sensorsNumber.setRange(1, self.NUM_SENSORS)
        self.sensorsNumber.setDecimals(0)
        self.sensorsNumber.setDisabled(True)
        sensors_val = self.cfg.getint("APPLICATION", "SensorsNumber")
        if not 1 <= sensors_val <=8: sensors_val = 8
        self.sensorsNumber.setValue(sensors_val) 
        
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
        smoothing_coef = self.cfg.getfloat("APPLICATION", "EnvelopeSmoothingCoefficient")
        if not 0.0 <= smoothing_coef <= 1.0:  smoothing_coef = 0.9
        self.envelopeSmoothingCoefficient.setValue(smoothing_coef)
        self.envelopeSmoothingCoefficient.setDisabled(not self.cfg.getboolean("APPLICATION", "Envelope"))
        self.EnvelopeSignalAction.toggled.connect(self.EnvelopeSignalActionTriggered)
        
        self.RMSsignalAction = QtWidgets.QCheckBox('RMS:', self)
        self.RMSsignalAction.setChecked(self.cfg.getboolean("APPLICATION", "RMS"))
        self.RMSsignalAction.toggled.connect(self.RMSsignalActionTriggered)
        self.RMSsignalAction1 = QtWidgets.QLabel('    ', self)
        self.RMSsignalAction2 = QtWidgets.QLabel('      ', self)
        self.RMSinterval = QtWidgets.QDoubleSpinBox()
        self.RMSinterval.setSingleStep(0.01)
        self.RMSinterval.setRange(0.01, 10)
        rms_val = self.cfg.getfloat("APPLICATION", "RMSinterval")
        if not 0.01 <= rms_val <= 10:  rms_val = 0.5
        self.RMSinterval.setValue(rms_val)
        self.RMSinterval.setDisabled(not self.cfg.getboolean("APPLICATION", "RMS"))
                
        self.bandstopAction = QtWidgets.QCheckBox('BANDSTOP FILTER:', self)
        self.bandstopAction.setChecked(self.cfg.getboolean("APPLICATION", "BandStopFilter"))
        self.bandstopAction.toggled.connect(self.bandstopActionTriggered)
        
        self.notchActiontypeBox=QtWidgets.QComboBox()
        self.notchActiontypeBox.addItem("50 Hz")
        self.notchActiontypeBox.addItem("60 Hz")
        self.notchActiontypeBox.setDisabled(not self.cfg.getboolean("APPLICATION", "BandStopFilter"))
                        
        self.bandpassAction = QtWidgets.QCheckBox('BANDPASS FILTER:', self)
        self.bandpassAction.setChecked(self.cfg.getboolean("APPLICATION", "BandPassFilter"))
        self.bandpassAction1 = QtWidgets.QLabel('  -  ', self)
        self.bandpassAction2 = QtWidgets.QLabel('       ', self)
        self.bandpassAction.toggled.connect(self.bandpassActionTriggered)
        
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
        
        pg.setConfigOptions(antialias=False) 
        self.pw = []

        for i in range(self.NUM_SENSORS):
            widget = CustomPlotWidget(sensors_spinbox=self.sensorsNumber)
            self.pw.append(widget)
            widget.setXLink(self.pw[0]) 

        
        # Plot widget for spectral Plot
        self.pwFFT = pg.PlotWidget(background=(13, 13, 13, 255))
        self.pwFFT.showGrid(x=True, y=True, alpha=0.3) 
        self.pFFT = self.pwFFT.plot()
        self.pFFT.setPen(color=(100, 255, 255), width=1)
        self.pwFFT.setLabel('bottom', 'Frequency', 'Hz')
                
        # Histogram widget
        self.pb = [] # Histogram item array, index - sensor number
        self.pbar = pg.PlotWidget(background=(13 , 13, 13, 255))
        self.pbar.showGrid(x=True, y=True, alpha=0.3)  
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
            trigger_val = self.cfg.getint(f"SENSOR{i+1}", "Trigger_value")
            if not 0 <= trigger_val <= 2500: trigger_val = 100
            self.TriggerValue[i].setValue(trigger_val)
            
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
        topleft.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        
        plotLayout = []
        self.row = []
        for i in range(self.NUM_SENSORS):
            plotLayout.append(QtWidgets.QGridLayout())
            if i % 2 == 0: plotLayout[i].addWidget(backLabel[i//2], 0, 0, 10, 1)
            plotLayout[i].addWidget(numberLabel[i], 0, 0, 10, 1, Qt.AlignmentFlag.AlignVCenter)
            plotLayout[i].addWidget(self.pw[i], 0, 1, 10, 50)
            plotLayout[i].addWidget(self.ChargeLabel[i], 0, 49) 
            plotLayout[i].addWidget(self.TriggerLabel[i], 1, 49) 
            plotLayout[i].addWidget(self.TriggerValue[i], 1, 50)   
            plotLayout[i].addWidget(self.NumberEMG_Lable[i], 2, 49) 
            plotLayout[i].addWidget(self.NumberEMG[i], 2, 50) 
            plotLayout[i].setContentsMargins(0, 0, 0, 0)    
            
            self.row.append(QtWidgets.QWidget())
            self.row[i].setLayout(plotLayout[i])
            
        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(1)

        for row in self.row[:self.NUM_SENSORS]: splitter.addWidget(row)

        layout = QtWidgets.QGridLayout()       
        layout.addWidget(splitter, 0, 0, 40, 4)
        layout.addWidget(self.pbar, 0, 4, 20, 11)
        layout.addWidget(self.pwFFT, 20, 4, 16, 11)
        layout.setColumnStretch(2, 2)

        layout.addWidget(self.sensorSelectedAction , 20, 13, 1, 1)
        layout.addWidget(self.sensorSelectedActionBox , 20, 14, 1, 1)  
        
        layout.addWidget(self.textWindow, 37, 4, 3, 12)  
        
        vbox.addLayout(layout)
        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)  
        self.showMaximized()
        self.show()    
        
        # Serial monitor
        self.serialMonitor = SerialMonitor(self.delay)
        
        existing_ports = {self.COMports.itemText(i) for i in range(self.COMports.count())}
        
        for port in self.serialMonitor.ports:
            if port not in existing_ports:
                self.COMports.addItem(port)
                    
        if self.serialMonitor.COM:
            self.serialMonitor.serialConnect()
            self.liveFromSerialAction.setChecked(True)
            self.dataRecordingAction.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live from " + self.serialMonitor.COM +" \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.COMports.setDisabled(True)
            self.refreshAction.setDisabled(False)

        self.sensorsNumber.valueChanged.connect(self.setSensorsNumber)        
        sensors_val = self.cfg.getint("APPLICATION", "SensorsNumber")
        if not 1 <= sensors_val <=8: sensors_val = 8
        self.setSensorsNumber(sensors_val)
        self.mainrun = MainRun(self.delay)
        self.mainrun.bufferUpdated.connect(self.updateListening, QtCore.Qt.ConnectionType.QueuedConnection)  
        print(">>> MYOblue_GUI was launched successfully.")        
    
    def bandstopActionTriggered(self):
        self.cfg.set("APPLICATION", "BandStopFilter", str(self.bandstopAction.isChecked()))
        if  self.bandstopAction.isChecked(): 
            self.notchActiontypeBox.setDisabled(False)
        else:
            self.notchActiontypeBox.setDisabled(True)
    
    def bandpassActionTriggered(self):
        self.cfg.set("APPLICATION", "BandPassFilter", str(self.bandpassAction.isChecked()))
        if self.bandpassAction.isChecked():
            self.passLowFreq.setDisabled(False)
            self.passHighFreq.setDisabled(False)
        else:
            self.passLowFreq.setDisabled(True)
            self.passHighFreq.setDisabled(True)
    
    def RMSsignalActionTriggered(self):
        self.cfg.set("APPLICATION", "RMS", str(self.RMSsignalAction.isChecked()))
        if self.RMSsignalAction.isChecked():
            self.RMSinterval.setDisabled(False)
        else:
            self.RMSinterval.setDisabled(True)
    
    def EnvelopeSignalActionTriggered(self):
        self.cfg.set("APPLICATION", "Envelope", str(self.EnvelopeSignalAction.isChecked()))
        if self.EnvelopeSignalAction.isChecked():
            self.envelopeSmoothingCoefficient.setDisabled(False)
            self.MovingAverage.MA_alpha = self.envelopeSmoothingCoefficient.value()
        else:
            self.envelopeSmoothingCoefficient.setDisabled(True)
    
    def rawSignalActionTriggered(self):
        self.cfg.set("APPLICATION", "RAW_EMG", str(self.rawSignalAction.isChecked()))
        if self.rawSignalAction.isChecked():
            self.rectificationSignalAction.setCheckState(False)
    
    def rectificationSignalActionTriggered(self):
        self.cfg.set("APPLICATION", "Rectification", str(self.rectificationSignalAction.isChecked()))
        if self.rectificationSignalAction.isChecked():
            self.rawSignalAction.setCheckState(Qt.CheckState.Unchecked)
    
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
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "pause ON" + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
        else:
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "pause OFF" + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)

    # Refresh data
    def refresh(self):
        self.l = [0] * self.NUM_SENSORS
        self.dataWidth = int((self.timeWidth + 2)*self.fs)
        self.data.refresh(self.dataWidth)
        self.msg_end = bytearray([0])      
        self.ms_len =  [0]*self.NUM_SENSORS
        self.MSG_NUM_0 = [0]*self.NUM_SENSORS
        self.slider.setValue(0)
        self.sliderpos = 0
        self.TIMER = 0
        self.FFT = np.zeros((self.NUM_SENSORS, 500), dtype=np.float32) 
        
        self.pll_initialized = [False] * self.NUM_SENSORS
        self.v_time = [0.0] * self.NUM_SENSORS
        
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

            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.recordingFileName_TXT = os.path.join(self.REC_DIR, timestamp + ".txt")
            self.recordingFileName_BIN = os.path.join(self.REC_DIR, timestamp + ".bin")

            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "recording to \"" + os.path.join(os.getcwd(), self.recordingFileName_BIN) + "\"\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.recordingFile_TXT = open(self.recordingFileName_TXT, "a") # Data file creation
            self.recordingFile_TXT.write(datetime.now().strftime("Date: %Y.%m.%d\rTime: %H:%M:%S") + "\r\n") # Data file name
            self.recordingFile_TXT.write("File format: \r\n8 sensors data in mkV\r\n") # Data file format
            self.recordingFile_BIN = open(self.recordingFileName_BIN, 'ab')
            self.is_recording = True
        else:
            if not self.PlaybackAction.isChecked():
                self.refreshAction.setDisabled(False)
            self.is_recording = False
            if getattr(self, 'recordingFile_TXT', None) is not None:
                self.recordingFile_TXT.close()
                self.recordingFile_TXT = None
            if getattr(self, 'recordingFile_BIN', None) is not None:
                self.recordingFile_BIN.close()
                self.recordingFile_BIN = None
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

        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a file',self.REC_DIR,
                                        'Binary Files (*.bin);;All Files (*)')
        if path != ('', ''):
            self.loadFileName = path[0]
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
      
    def keyPressEvent(self, event):
        digit_char = event.text()
        
        if digit_char and digit_char.isdigit() and len(digit_char) == 1 and digit_char != '0':
            exercise_start_timestamp = time.perf_counter() - self.TIMER
            self.markers_list.append((digit_char, exercise_start_timestamp))
            
            if hasattr(self, 'pw'):
                for pw in self.pw:
                    marker_line = pg.InfiniteLine(
                        pos=exercise_start_timestamp, 
                        pen=pg.mkPen(color='w', width=1, style=QtCore.Qt.DashLine),
                        label=f"Ex {digit_char}",
                        labelOpts={'position': 0.9, 'color': 'w'}
                    )
                    
                    pw.addItem(marker_line)                
        super().keyPressEvent(event)        

    # Update
    def updateListening(self):  
        
        raw_enabled = self.rawSignalAction.isChecked()
        rect_enabled = self.rectificationSignalAction.isChecked()
        env_enabled = self.EnvelopeSignalAction.isChecked()
        rms_enabled = self.RMSsignalAction.isChecked()
        bandstop_enabled = self.bandstopAction.isChecked()
        bandpass_enabled = self.bandpassAction.isChecked()
        notch_type = self.notchActiontypeBox.currentText()
        num_sensors = int(self.sensorsNumber.value())
        rms_interval = self.RMSinterval.value()
        
        self.cfg.set('APPLICATION', 'EnvelopeSmoothingCoefficient', str(self.envelopeSmoothingCoefficient.value()))
        self.cfg.set('APPLICATION', 'RMSinterval', str(rms_interval))
        self.cfg.set("APPLICATION", "BandPassFilterLF", str(self.passLowFreq.value()))
        self.cfg.set("APPLICATION", "BandPassFilterHF", str(self.passHighFreq.value()))
        
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
        
        # Read data from File               
        if (self.PlaybackAction.isChecked() and self.loadFileName != ''):
            self.readFromFile()
        
        # Read data from serial          
        if (self.liveFromSerialAction.isChecked()):
            self.readFromSerial()

        while self.sensorSelectedActionBox.count() < num_sensors: 
            self.sensorSelectedActionBox.addItem(str(self.sensorSelectedActionBox.count() + 1))
            
        while self.sensorSelectedActionBox.count() > num_sensors: 
            self.sensorSelectedActionBox.removeItem(self.sensorSelectedActionBox.count()-1)
            
        max_time = max(self.data.time[i][self.l[i] - 1] for i in range(num_sensors))
        start = self.timeWidth * (max_time // self.timeWidth)
        end = start + self.timeWidth
        n = int(rms_interval * 1000 / 2)
    
        if not self.pauseAction.isChecked():
            self.pw[0].setXRange(start, end)   
            
        if (self.PlaybackAction.isChecked() and self.loadFileName != '') or (self.liveFromSerialAction.isChecked()):  
            max_time = 0
            
            if not hasattr(self, '_fft_frame_counter'):
                self._fft_frame_counter = 0
            self._fft_frame_counter += 1

            for i in range( num_sensors ):
                self.cfg.set(f"SENSOR{i+1}", "dt_(s)", str(self.dt[i]))
                self.cfg.set(f"SENSOR{i+1}", "Trigger_value", str(self.TriggerValue[i].value()))
                
                pw = self.pw[i]
                dt = self.dt[i]
                ms_len = self.ms_len[i]
                
                plot = np.concatenate((self.data.raw[i][self.l[i]:], self.data.raw[i][:self.l[i]]))
                timePlot = np.concatenate((self.data.time[i][self.l[i]:], self.data.time[i][:self.l[i]]))


                plot -= 8192
                plot *= 0.30517578125  # Precomputed constant (2.5 / 16384.0 * 2000)
            
                if  bandstop_enabled:
                    if (notch_type == "50 Hz"): plot = self.bandstop_filter_50Hz.apply(plot, 1/dt)
                    if (notch_type == "60 Hz"): plot = self.bandstop_filter_60Hz.apply(plot, 1/dt)
                                
                if bandpass_enabled: 
                    plot = self.bandpass_filter.apply(plot, self.passLowFreq.value(), self.passHighFreq.value(), 1/dt)
                    self.data.rectification[i] = abs(plot)
                else: self.data.rectification[i] = abs(self.HP_filter.apply(plot, 1, 1/dt))
                
                if  bandstop_enabled or bandpass_enabled: plot[0:int(1.5*self.fs)] = 0
                self.data.rectification[i][0:int(1.5*self.fs)] = 0
                self.data.envelope[i][0:int(1.5*self.fs)] = 0
                self.data.RMS[i][0:int(1.5*self.fs)] = 0
                
                if not self.pauseAction.isChecked():
                    
                    # Plot raw data or rectification
                    if  raw_enabled: pw.p.setData(y=plot, x=timePlot)
                    elif  rect_enabled: pw.p.setData(y=self.data.rectification[i], x=timePlot)
                    elif not raw_enabled: pw.p.clear()
                    
                    # Plot envelope data
                    if  env_enabled: pw.pe.setData(y=self.data.envelope[i], x=timePlot)
                    else: pw.pe.clear()     
                    
                    # Plot RMS data
                    if  rms_enabled: pw.pi.setData(y=self.data.RMS[i], x=timePlot)
                    else: pw.pi.clear()
                    
                    # Plot histogram
                    self.pb[i].setOpts(height=2*self.data.RMS[i][-1])
                    
                self.data.plot[i] = plot
                self.data.timePlot[i] = timePlot
   
                if ms_len > 0:
                    self.data.envelope[i] = np.concatenate((self.data.envelope[i][ms_len:], self.data.envelope[i][:ms_len]))
                    self.data.RMS[i] = np.concatenate((self.data.RMS[i][ms_len:], self.data.RMS[i][:ms_len]))

                    trigger_val = self.TriggerValue[i].value()
                    emg_counter = self.NumberEMG[i].value()
                    current_flag = self.FlagEMG[i]
    
                    for j in range (self.dataWidth - ms_len, self.dataWidth):
                        self.data.envelope[i][j] = self.MovingAverage.movingAverage(i, self.data.rectification[i][j])
                        
                        if j >= n + 1:
                            I1 = (self.data.envelope[i][j-n]**2 + self.data.envelope[i][j-n-1]**2)*dt*0.5
                            I2 = (self.data.envelope[i][j]**2 + self.data.envelope[i][j-1]**2)*dt*0.5
                            self.data.RMS[i][j] = abs((self.data.RMS[i][j-1]**2 + (I2 - I1)/rms_interval))**0.5
                        else:
                            self.data.RMS[i][j] = 0
                            
                        if (current_flag == 0) & (self.data.RMS[i][j] >= trigger_val):
                            current_flag = 1
                            self.NumberEMG[i].setValue(emg_counter + 1)
                        
                        if (current_flag == 1) & (self.data.RMS[i][j] < trigger_val):
                            current_flag = 0
            
            # Plot FFT data
            i = self.sensorSelectedActionBox.currentIndex()
            data_segment = self.data.plot[i][-500:]
            Y = np.abs(fft(data_segment)) / 500
            self.FFT[i] = 0.5 * self.FFT[i] + 0.5 * Y
            
            if not self.pauseAction.isChecked() and (self._fft_frame_counter % 2 == 0):
                X = np.linspace(0, 1 / dt, 500)
                half = 250
                self.pFFT.setData(x=X[2:half], y=self.FFT[i][2:half])

            if not self.pauseAction.isChecked() and hasattr(self, 'pw') and len(self.pw) > 0:
                left_view_limit = start 
                
                for widget in self.pw:
                    for item in list(widget.plotItem.items):
                        if isinstance(item, pg.InfiniteLine):
                            if item.value() < left_view_limit:
                                widget.removeItem(item)

            if (self.dataRecordingAction.isChecked()):
                max_ms_len = max(self.ms_len)
                DataRec = np.zeros((self.NUM_SENSORS, max_ms_len), dtype=np.float32)
                DataRecBin = np.zeros((self.NUM_SENSORS, max_ms_len), dtype=np.float32)
                TimeRec = np.zeros((self.NUM_SENSORS, max_ms_len), dtype=np.float64)
                flag = 0
                
                Data = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
                Time = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float64)
                
                for i in range( num_sensors ): 
                    Data[i] = np.roll(self.data.raw[i], -self.l[i])
                    Time[i] = np.roll(self.data.time[i], -self.l[i])
                
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
                    if (self.num[i] >= 0 ) and (self.num[i] <= self.dataWidth - max_ms_len) and self.Fl == 1:
                        DataRec[i] = self.data.plot[i][self.num[i]: self.num[i] + max_ms_len]   
                        DataRecBin[i] = Data[i][self.num[i]: self.num[i] + max_ms_len] 
                        TimeRec[i] = Time[i][self.num[i]: self.num[i] + max_ms_len] 
                        self.num[i] += max_ms_len 
                        flag = 1
                        
                if flag == 1:
                    for i in range(max_ms_len):
                        max_time_at_i = max(TimeRec[sensor_idx][i] for sensor_idx in range(self.NUM_SENSORS))
                        marker_key_char = '0'
                        
                        for marker in self.markers_list[:]:
                            key_char, marker_time = marker
                            time_differences = abs(max_time_at_i - marker_time)

                            if time_differences < 0.001:
                                marker_key_char = key_char
                                self.markers_list.remove(marker)
                        
                        sensors_data = str(round(DataRec[0][i]))
                        for j in range(1, self.NUM_SENSORS): sensors_data += (" " + str(round(DataRec[j][i])))
                        sensors_data += " " + marker_key_char + '\n'
                        self.recordingFile_TXT.write(sensors_data)
                        
                        bin_data = struct.pack("H H H H H H H H", int(DataRecBin[0][i]), int(DataRecBin[1][i]), int(DataRecBin[2][i]), int(DataRecBin[3][i]), 
                                               int(DataRecBin[4][i]), int(DataRecBin[5][i]), int(DataRecBin[6][i]), int(DataRecBin[7][i]))             
                        self.recordingFile_BIN.write(bin_data)             

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
                burst_counters_0 = [0] * self.NUM_SENSORS
                burst_counters = [0] * self.NUM_SENSORS
                for burst_i in range(0, len(msg), 246):
                    s_num = int(msg[burst_i+2])-1
                    if 0 <= s_num < self.NUM_SENSORS:
                        burst_counters[s_num] += 1
                
                burst_counters_0 = burst_counters        
                for msg_i in range(0, len(msg), 246):
                    sensorNum = int(msg[msg_i+2])-1 
                    if not 0 <= sensorNum <= 7: break
                    MSG_NUM = int(msg[msg_i+3] | msg[msg_i+4] << 8 | msg[msg_i+5] << 16)
                    
                    if self.MSG_NUM_0[sensorNum] == 0: 
                        if self.TIMER == 0:
                            self.TIMER = TIME 
                            
                        self.MSG_NUM_0[sensorNum] = MSG_NUM
                        self.data.time[sensorNum][self.l[sensorNum]-1] = TIME - self.TIMER
                        
                        self.pll_initialized[sensorNum] = True
                        self.v_time[sensorNum] = max(self.v_time) #TIME - self.TIMER
                        self.sensor_uptime[sensorNum] = self.v_time[sensorNum]
                            
                    if MSG_NUM - self.MSG_NUM_0[sensorNum] > 0:
                        self.data.time[sensorNum][self.l[sensorNum]-1] += self.dt[sensorNum]*119*(MSG_NUM - self.MSG_NUM_0[sensorNum] - 1)
                        
                        if MSG_NUM - self.MSG_NUM_0[sensorNum] == 1:
                            self.v_time[sensorNum] += self.dt[sensorNum] * 119
                        else:
                            self.v_time[sensorNum] += (MSG_NUM - self.MSG_NUM_0[sensorNum]) * self.dt[sensorNum] * 119
                            
                        self.MSG_NUM_0[sensorNum] = MSG_NUM
                    else:
                        self.MSG_NUM_0[sensorNum] = MSG_NUM
                        self.data.time[sensorNum][self.l[sensorNum]-1] = TIME - self.TIMER
                        
                        self.v_time[sensorNum] = max(self.v_time)
                        self.sensor_uptime[sensorNum] = self.v_time[sensorNum]

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
                        time_pc = TIME - self.TIMER
                        error = time_pc - self.v_time[sensorNum]
                        burst_counters[sensorNum] -= 1
                    
                        if burst_counters_0[sensorNum] == self.delay // 119 and burst_counters[sensorNum] == 0 and error*1000 < 130:
                            error = time_pc - self.v_time[sensorNum]
                            if self.v_time[sensorNum] - self.sensor_uptime[sensorNum] < 5.0: self.v_time[sensorNum] += 0.2 * error
                            elif self.v_time[sensorNum] - self.sensor_uptime[sensorNum] < 10.0:  self.v_time[sensorNum] += 0.05 * error
                            elif self.v_time[sensorNum] - self.sensor_uptime[sensorNum] < 15.0: self.v_time[sensorNum] += 0.02 * error
                                                 
                            self.dt[sensorNum] += (0.0001 * error) / 119
                            if self.dt[sensorNum] > 0.001015:  self.dt[sensorNum] = 0.001015
                            if self.dt[sensorNum] < 0.000985:  self.dt[sensorNum] = 0.000985

                        sample_idx = 0
                        for i in range(msg_i+8, msg_i + 246, 2):
                            if (self.l[sensorNum] == self.dataWidth):
                                self.l[sensorNum] = 0 
                                if (self.dataRecordingAction.isChecked()):
                                    self.recordingFile_BIN.close()
                                    self.recordingFile_TXT.close()
                                    self.recordingFile_BIN = open(self.recordingFileName_BIN, 'ab')
                                    self.recordingFile_TXT = open(self.recordingFileName_TXT, "a")
                                    
                            self.data.raw[sensorNum][self.l[sensorNum]] = int(msg[i] | msg[i+1] << 8)
                            if ( self.l[sensorNum] > 0):
                                self.data.time[sensorNum][self.l[sensorNum]] = self.data.time[sensorNum][self.l[sensorNum] - 1] + self.dt[sensorNum] 
                            else:
                                self.data.time[sensorNum][self.l[sensorNum]] = self.data.time[sensorNum][self.dataWidth - 1] + self.dt[sensorNum]
                    
                            self.l[sensorNum] += 1
                            if (self.ms_len[sensorNum] < self.dataWidth): 
                                self.ms_len[sensorNum] += 1 
                            sample_idx += 1
                            
                        timeDifference = self.v_time[sensorNum] - self.data.time[sensorNum][self.l[sensorNum]-1]
                        if self.v_time[sensorNum] - self.sensor_uptime[sensorNum] < 5.0: accuracy = 0.1
                        elif self.v_time[sensorNum] - self.sensor_uptime[sensorNum] < 10.0: accuracy = 0.05
                        elif self.v_time[sensorNum] - self.sensor_uptime[sensorNum] < 15.0: accuracy = 0.02
                        elif self.v_time[sensorNum] - self.sensor_uptime[sensorNum] < 30.0: accuracy = 0.005
                        else: accuracy = 0.000001
                        
                        if self.v_time[sensorNum] - self.sensor_uptime[sensorNum] < 30:
                            if timeDifference > accuracy:
                                self.data.time[sensorNum][:] += timeDifference
                            if timeDifference < -accuracy:
                                self.data.time[sensorNum][:] += timeDifference 

    def setSensorsNumber(self, num):
        
        self.cfg.set('APPLICATION', 'SensorsNumber', str(int(num)))
        with open(os.path.join(self.BASE_DIR, "config.ini"), "w", encoding="utf-8") as f:
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
            with open(os.path.join(self.BASE_DIR, "config.ini"), "w", encoding="utf-8") as f:
                self.cfg.write(f)
                
            if hasattr(self, 'is_recording') and self.is_recording:
                self.is_recording = False
                
                if getattr(self, 'recordingFile_TXT', None) is not None:
                    try:
                        self.recordingFile_TXT.flush()
                        self.recordingFile_TXT.close()
                    except Exception:
                        pass
                        
                if getattr(self, 'recordingFile_BIN', None) is not None:
                    try:
                        self.recordingFile_BIN.flush()
                        self.recordingFile_BIN.close()
                    except Exception:
                        pass
    
            self.mainrun.running = False
            self.serialMonitor.serialDisconnection()
            event.accept()

class Data:
    def __init__(self, NUM_SENSORS, dataWidth):
        self.dataWidth = dataWidth
        self.NUM_SENSORS = NUM_SENSORS
        self.raw = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32) 
        self.plot = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
        self.envelope = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
        self.RMS = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
        self.rectification = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
        self.time = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32) 
        self.timePlot = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
    def refresh(self, dataWidth):
        self.dataWidth = dataWidth
        self.raw = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32) 
        self.plot = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
        self.envelope = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32) 
        self.RMS = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
        self.rectification = np.zeros((self.NUM_SENSORS, self.dataWidth), dtype=np.float32)
        self.time = np.zeros((self.NUM_SENSORS, self.dataWidth)) 
        self.timePlot = np.zeros((self.NUM_SENSORS, self.dataWidth))

# Butterworth bandpass filter
class bandpass_filter:
    def __init__(self, lowcut, highcut, fs):
        self.order = 4
        self.fs = fs
        self.lowcut_hz = lowcut
        self.highcut_hz = highcut
        nyq_low = lowcut / (0.5 * fs)
        nyq_high = highcut / (0.5 * fs)
        self.b, self.a = butter(self.order, [nyq_low, nyq_high], btype='bandpass')
        
    def apply(self, data, lowcut, highcut, fs):
        if self.lowcut_hz != lowcut or self.highcut_hz != highcut or self.fs != fs:
            self.fs = fs
            self.lowcut_hz = lowcut
            self.highcut_hz = highcut
            nyq_low = lowcut / (0.5 * fs)
            nyq_high = highcut / (0.5 * fs)
            self.b, self.a = butter(self.order, [nyq_low, nyq_high], btype='bandpass')
        return lfilter(self.b, self.a, data)

# Butterworth bandstop filter
class bandstop_filter_50Hz:
    def __init__(self, fs):
        self.order = 4
        self.fs = fs
        self.b = [None] * 4
        self.a = [None] * 4
        self._compute_coefficients()
            
    def _compute_coefficients(self):
        nyq = 0.5 * self.fs
        for i in range(4):
            lowcut = (48 + 50 * i) / nyq
            highcut = (52 + 50 * i) / nyq
            self.b[i], self.a[i] = butter(self.order, [lowcut, highcut], btype='bandstop')

    def apply(self, data, fs):
        if self.fs != fs:
            self.fs = fs
            self._compute_coefficients()
        for i in range(4):
            data = lfilter(self.b[i], self.a[i], data)
        return data

# Butterworth bandstop filter
class bandstop_filter_60Hz:
    def __init__(self, fs):
        self.order = 4
        self.fs = fs
        self.b = [None] * 4
        self.a = [None] * 4
        self._compute_coefficients()
            
    def _compute_coefficients(self):
        nyq = 0.5 * self.fs
        for i in range(4):
            lowcut = (58 + 60 * i) / nyq
            highcut = (62 + 60 * i) / nyq
            self.b[i], self.a[i] = butter(self.order, [lowcut, highcut], btype='bandstop')

    def apply(self, data, fs):
        if self.fs != fs:
            self.fs = fs
            self._compute_coefficients()
        for i in range(4):
            data = lfilter(self.b[i], self.a[i], data)
        return data

# Butterworth bandpass filter
class HP_filter:
    def __init__ (self, lowcut, fs):
        self.order = 4
        self.fs = fs
        self.lowcut_hz = lowcut
        self.nyq_lowcut = lowcut / (0.5 * fs)
        self.b, self.a = butter(self.order, self.nyq_lowcut, btype='highpass')
        
    def apply(self, data, lowcut, fs):
        if self.lowcut_hz != lowcut or self.fs != fs:
            self.fs = fs
            self.lowcut_hz = lowcut
            self.nyq_lowcut = lowcut / (0.5 * fs)
            self.b, self.a = butter(self.order, self.nyq_lowcut, btype='highpass')
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


class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_x_val = None  
        
        self.normal_pen = QPen(QColor(150, 150, 150)) 
        self.mouse_pen = QPen(QColor(230, 230, 230))  

    def setMouseValue(self, val):
        if val is not None and self.mouse_x_val is not None:
            if abs(val - self.mouse_x_val) < 0.0001: 
                return
                
        self.mouse_x_val = val
        self.picture = None  
        self.update()

    def tickValues(self, minVal, maxVal, size):
        ticks = super().tickValues(minVal, maxVal, size)
        if self.mouse_x_val is not None and minVal <= self.mouse_x_val <= maxVal and ticks:
            new_ticks = []
            for idx, (spacing, t_list) in enumerate(ticks):
                if idx == 0:
                    new_list = list(t_list)
                    new_list.append(self.mouse_x_val)
                    new_ticks.append((spacing, new_list))
                else:
                    new_ticks.append((spacing, t_list))
            return new_ticks
        return ticks
        
    def tickStrings(self, values, scale, spacing):
        strings = []
        collision_threshold = spacing * 0.2
        
        for val in values:
            if val < 0:
                strings.append("")
                continue
                
            if self.mouse_x_val is not None and abs(val - self.mouse_x_val) < 1e-5:
                minutes = int(val // 60)
                seconds = int(val % 60)
                milliseconds = int((val % 1) * 1000)
                strings.append(f"[{minutes:02d}:{seconds:02d}.{milliseconds:03d}]")
            else:
                if self.mouse_x_val is not None and abs(val - self.mouse_x_val) < collision_threshold:
                    strings.append("") 
                else:
                    minutes = int(val // 60)
                    seconds = int(val % 60)
                    strings.append(f"{minutes:02d}:{seconds:02d}")
        return strings

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        for rect, flags, text in textSpecs:
            if "[" in text and "]" in text:
                p.setPen(self.mouse_pen)  
                p.drawText(rect, flags, text)
            else:
                p.setPen(self.normal_pen)  
                p.drawText(rect, flags, text)
                
        super().drawPicture(p, axisSpec, tickSpecs, [])

class LiveYAxisItem(pg.AxisItem):
    def __init__(self, sensors_spinbox, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_y_val = None  
        self.sensors_spinbox = sensors_spinbox  
        
        self.normal_pen = QPen(QColor(150, 150, 150)) 
        self.mouse_pen = QPen(QColor(230, 230, 230))   

    def setMouseValue(self, val):
        if val is not None and self.mouse_y_val is not None:
            if abs(val - self.mouse_y_val) < 0.01:
                return
        if val is None and self.mouse_y_val is None:
            return
        self.mouse_y_val = val
        self.picture = None  
        self.update()

    def tickValues(self, minVal, maxVal, size):
        ticks = super().tickValues(minVal, maxVal, size)
        if self.mouse_y_val is not None and minVal <= self.mouse_y_val <= maxVal and ticks:
            new_ticks = []
            for idx, (spacing, t_list) in enumerate(ticks):
                if idx == 0:
                    new_list = list(t_list)
                    new_list.append(self.mouse_y_val)
                    new_ticks.append((spacing, new_list))
                else:
                    new_ticks.append((spacing, t_list))
            return new_ticks
        return ticks

    def tickStrings(self, values, scale, spacing):
        strings = []
        current_sensors = int(self.sensors_spinbox.value())
        collision_threshold = spacing * (0.05 + current_sensors/20)  
        
        for val in values:
            if self.mouse_y_val is not None and abs(val - self.mouse_y_val) < 0.01:
                r_val = round(val, 0)
                strings.append(f"{r_val:.0f}")
            else:
                if self.mouse_y_val is not None and abs(val - self.mouse_y_val) < collision_threshold:
                    strings.append("")  
                else:
                    try:
                        strings.append(str(int(round(val))))
                    except (ValueError, TypeError):
                        strings.append(str(val))
        return strings

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        for rect, flags, text in textSpecs:
            if self.mouse_y_val is not None and text == f"{round(self.mouse_y_val, 0):.0f}":
                p.setPen(self.mouse_pen)  
                p.drawText(rect, flags, text)
            else:
                p.setPen(self.normal_pen) 
                p.drawText(rect, flags, text)
                
        super().drawPicture(p, axisSpec, tickSpecs, [])
    
    
class CustomPlotWidget(pg.PlotWidget):
    mouse_moved_signal = pyqtSignal(int, float, str)
    
    def __init__(self, parent=None, sensors_spinbox=None, **kwargs):
        sensors_spinbox = kwargs.pop('sensors_spinbox', sensors_spinbox)
        
        if 'axisItems' not in kwargs:
            kwargs['axisItems'] = {
                'bottom': TimeAxisItem(orientation='bottom'),
                'left': LiveYAxisItem(sensors_spinbox=sensors_spinbox, orientation='left')
            }
        super().__init__(parent, **kwargs)
        self.plotItem.disableAutoRange(pg.ViewBox.YAxis) 
        self.plotItem.setYRange(-2000, 2000)
        
        self.setBackground(background=(21, 21, 21, 255))
        self.getAxis('left').setWidth(40)
        self.showGrid(x=True, y=True, alpha=0.3) 
        self.getAxis('bottom').setStyle(showValues=False)
        
        self.p = self.plot(skipFiniteCheck=True)
        self.pe = self.plot(skipFiniteCheck=True)
        self.pi = self.plot(skipFiniteCheck=True)
        
        self.p.setPen(color=(100, 255, 255), width=1)
        self.pe.setPen(color=(255, 0, 0), width=1)
        self.pi.setPen(color=(0, 255, 0), width=1)
        
        self.p.setClipToView(True)
        self.pe.setClipToView(True)
        self.pi.setClipToView(True)
        
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=10, slot=self.onMouseMove)
            
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)
       
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.plotItem.vb.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)

    def onMouseMove(self, evt):
        pos = evt[0]
        main_win = self.window()
        
        if not self.sceneBoundingRect().contains(pos):
            self.getAxis('left').setMouseValue(None)
            if hasattr(main_win, 'pw'):
                visible_axis_widget = getattr(main_win, 'sensorsNumber', None)
                visible_idx = int(visible_axis_widget.value()) - 1 if visible_axis_widget else len(main_win.pw) - 1
                if 0 <= visible_idx < len(main_win.pw):
                    main_win.pw[visible_idx].getAxis('bottom').setMouseValue(None)                
            if hasattr(main_win, 'v_lines'):
                for line in main_win.v_lines: line.hide()
            if hasattr(main_win, 'h_lines'):
                for line in main_win.h_lines: line.hide()
            return
        
        vb = self.plotItem.vb
        mp = vb.mapSceneToView(pos)
        
        x_val = mp.x()
        minutes = int(max(0, x_val) // 60)
        seconds = int(max(0, x_val) % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        if hasattr(main_win, 'v_lines'):
            for line in main_win.v_lines:
                line.setPos(mp.x())
                line.show()
                
        if hasattr(main_win, 'pw'):
            for idx, widget in enumerate(main_win.pw):
                if widget == self:
                    if hasattr(main_win, 'h_lines') and len(main_win.h_lines) > idx:
                        main_win.h_lines[idx].setPos(mp.y())
                        main_win.h_lines[idx].show()
                    
                    self.getAxis('left').setMouseValue(mp.y())
                    
                    for i in range(0, 8):
                        last_widget = main_win.pw[i]
                        last_widget.getAxis('bottom').setMouseValue(mp.x())
                    
                    self.mouse_moved_signal.emit(idx + 1, mp.y(), time_str)
                else:
                    if hasattr(main_win, 'h_lines') and len(main_win.h_lines) > idx:
                        main_win.h_lines[idx].hide()
                    widget.getAxis('left').setMouseValue(None)

class ClampedSpinBox(QtWidgets.QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)

    def validate(self, text, pos):
        # Always allow typing any digits
        return (QtGui.QValidator.Acceptable, text, pos)

    def valueFromText(self, text):
        # Convert text to int safely
        try:
            value = int(text)
        except ValueError:
            return self.minimum()

        # Clamp to range
        if value < self.minimum():
            return self.minimum()
        if value > self.maximum():
            return self.maximum()
        return value
    
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
                    self.ser = serial.Serial(
                        self.COM, 
                        self.baudRate, 
                        dsrdtr=False, 
                        rtscts=False, 
                        timeout=0
                    )
                    self.connect = True             
                    QtCore.QTimer.singleShot(500, self.ser.flushInput)
                except SerialException :
                    self.connect = False
                    
    def serialDisconnection(self):
        self.ser.close()
        self.connect = False
        
    def serialRead(self):  
        if not self.ser or not self.ser.is_open:
            return bytes(0)

        msg = bytes(0)
        try:
            if self.ser.in_waiting > 0:
                msg = self.ser.read(self.ser.in_waiting)
        except (SerialException, OSError, AttributeError):
            try:
               self.ser.close()
               self.ser.open()
            except (SerialException, OSError):
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
    
    window.raise_()  
    window.activateWindow()
    window.setFocus()
    
    window.start()
    sys.exit(app.exec())
