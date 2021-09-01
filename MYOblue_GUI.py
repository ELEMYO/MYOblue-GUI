# Graphical interface for signal visualization and interaction with ELEMYO MYOblue sensors
# 2021-09-01 by ELEMYO (https://github.com/ELEMYO)
# 
# Changelog:
#     2021-09-01 - initial release

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

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import sys
import serial
import pyqtgraph as pg
import numpy as np
import time
from scipy.signal import butter, lfilter
import serial.tools.list_ports
from scipy.fftpack import fft

# Main window
class GUI(QtWidgets.QMainWindow):
    # Initialize constructor
    def __init__(self):
          super(GUI, self).__init__()
          self.initUI()
    # Custom constructor
    def initUI(self): 
        # Values
        COM = '' # Example: COM='COM6'
        baudRate = 1000000 # Serial frequency
        self.delay = 0.21 # Delay for graphic update
               
        self.setWindowTitle("MYOblue GUI v1.0.1 | ELEMYO" + "    ( COM Port not found )")
        self.setWindowIcon(QtGui.QIcon('img/icon.png'))
        self.l = [0]*4 # Current sensor data point
        self.dt = 0.002 # Time between two signal measurements in s
        self.fs = 1/self.dt # Signal discretization frequency in Hz
        self.passLowFrec = 10 # Low frequency for passband filter
        self.passHighFrec = 200 # Low frequency for passband filter
        self.dataWidth = int(12.2/self.dt) # Maximum count of ploting data points (6.2 secondes vindow)
        self.Time = np.zeros((4, self.dataWidth)) # Time array
        self.timeWidth = 10 # Time width of plot
        self.Data = np.zeros((4, self.dataWidth)) # Raw data matrix, first index - sensor number, second index - sensor data 
        self.DataEnvelope = np.zeros((4, self.dataWidth)) # Envelope of row data, first index - sensor number, second index - sensor data 
        
        # Accessory variables for envelope (for moving average method)
        self.MA = np.zeros((4, 3)) 
        self.MA_alpha = 0.95
        self.Y0 = np.zeros(4)
        self.X0 = np.zeros(4)
        
        # Accessory variables for data read from serial
        self.ms_len = [0]*4;
        self.msg_end = bytearray([0])
        
        self.FFT = np.zeros((4, 500)) # Fast Fourier transform data
        
        self.VDD = [0]*4
        self.MSG_NUM = [0]*4

        # Menu panel
        stopAction = QtGui.QAction(QtGui.QIcon('img/pause.png'), 'Stop/Start (Space)', self)
        stopAction.setShortcut('Space')
        stopAction.triggered.connect(self.stop)
        refreshAction = QtGui.QAction(QtGui.QIcon('img/refresh.png'), 'Refresh (R)', self)
        refreshAction.setShortcut('r')
        refreshAction.triggered.connect(self.refresh)
        exitAction = QtGui.QAction(QtGui.QIcon('img/out.png'), 'Exit (Esc)', self)
        exitAction.setShortcut('Esc')
        exitAction.triggered.connect(self.close)
        
        # Toolbar
        toolbar = self.addToolBar('Tool')
        toolbar.addAction(stopAction)
        toolbar.addAction(refreshAction)
        toolbar.addAction(exitAction)
        
        # Plot widgets for 1-4 sensor
        self.pw = [] # Plot widget array, index - sensor number
        self.p = [] # Raw data plot, index - sensor number
        self.pe = [] # Envelope data plot, index - sensor number
        for i in range(4):
            self.pw.append(pg.PlotWidget(background=(21 , 21, 21, 255)))
            self.pw[i].showGrid(x=True, y=True, alpha=0.7) 
            self.p.append(self.pw[i].plot())
            self.pe.append(self.pw[i].plot())
            self.p[i].setPen(color=(100, 255, 255), width=0.8)
            self.pe[i].setPen(color=(255, 0, 0), width=1)
        
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
        self.pb.append(pg.BarGraphItem(x=np.linspace(1, 2, num=1), height=np.linspace(1, 2, num=1), width=0.3, pen=QtGui.QColor(153, 0, 0), brush=QtGui.QColor(153, 0, 0)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(2, 3, num=1), height=np.linspace(2, 3, num=1), width=0.3, pen=QtGui.QColor(229, 104, 19), brush=QtGui.QColor(229, 104, 19)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(3, 4, num=1), height=np.linspace(3, 4, num=1), width=0.3, pen=QtGui.QColor(221, 180, 10), brush=QtGui.QColor(221, 180, 10)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(4, 5, num=1), height=np.linspace(4, 5, num=1), width=0.3, pen=QtGui.QColor(30, 180, 30), brush=QtGui.QColor(30, 180, 30)))
        for i in range(4):
            self.pbar.addItem(self.pb[i])  
        self.pbar.setLabel('bottom', 'Sensor number')
        
        # Styles
        centralStyle = "color: rgb(255, 255, 255); background-color: rgb(13, 13, 13);"
        editStyle = "border-style: solid; border-width: 1px;"
        
        # Settings zone
        filtersText = QtWidgets.QLabel("FILTERS:")
        self.passLowFreq = QtWidgets.QLineEdit(str(self.passLowFrec), self)
        self.passLowFreq.setMaximumWidth(30)
        self.passLowFreq.setStyleSheet(editStyle)
        self.passHighFreq = QtWidgets.QLineEdit(str(self.passHighFrec), self)
        self.passHighFreq.setMaximumWidth(30)
        self.passHighFreq.setStyleSheet(editStyle)
        self.bandpass = QtWidgets.QCheckBox("BANDPASS FILTER:")
        self.bandstop50 = QtWidgets.QCheckBox("NOTCH 50 Hz")
        self.bandstop60 = QtWidgets.QCheckBox("NOTCH 60 Hz")
        
        plotStyle = QtWidgets.QLabel("PLOT STYLE: ")
        self.signal = QtWidgets.QCheckBox("Signal")
        self.envelope = QtWidgets.QCheckBox("Envelope")
        self.signal.setChecked(True)
        
        
        self.envelopeSmoothing = QtWidgets.QLabel(" Envelope smoothing:")
        self.envelopeSmoothingСoefficient = QtWidgets.QLineEdit(str(self.MA_alpha), self)
        self.envelopeSmoothingСoefficient.setMaximumWidth(30)
        self.envelopeSmoothingСoefficient.setStyleSheet(editStyle)
        
        # Buttons for selecting sensor for FFT analysis
        fftButton = []
        for i in range(4):
            fftButton.append(QtWidgets.QRadioButton(str(i + 1)))
            fftButton[i].Value = i + 1
        fftButton[0].setChecked(True)
        self.button_group = QtWidgets.QButtonGroup()
        for i in range(4):
            self.button_group.addButton(fftButton[i], i + 1)
        self.button_group.buttonClicked.connect(self._on_radio_button_clicked)
        
        
        # Numbering of graphs
        backLabel = []
        for i in range(2):
            backLabel.append(QtWidgets.QLabel(""))
            backLabel[i].setStyleSheet("font-size: 25px; background-color: rgb(21, 21, 21);")
        
        numberLabel = []
        for i in range(4):
            numberLabel.append(QtWidgets.QLabel(" " + str(i+1) + " "))
        numberLabel[0].setStyleSheet("font-size: 25px; background-color: rgb(153, 0, 0); border-radius: 14px;")
        numberLabel[1].setStyleSheet("font-size: 25px; background-color: rgb(229, 104, 19); border-radius: 14px;") 
        numberLabel[2].setStyleSheet("font-size: 25px; background-color: rgb(221, 180, 10); border-radius: 14px;")
        numberLabel[3].setStyleSheet("font-size: 25px; background-color: rgb(30, 180, 30); border-radius: 14px;")
        
        
        self.ChargeLabel  = []
        for i in range(4):
            self.ChargeLabel.append(QtWidgets.QLabel("BATTERY CHARGE: 0.00"))
        
        # Main widget
        centralWidget = QtWidgets.QWidget()
        centralWidget.setStyleSheet(centralStyle)
        
        # Layout
        vbox = QtWidgets.QVBoxLayout()
        
        layout = QtWidgets.QGridLayout()
        layout.addWidget(backLabel[0], 0, 1, 10, 1)
        layout.addWidget(numberLabel[0], 0, 1, 10, 1, Qt.AlignVCenter)
        layout.addWidget(numberLabel[1], 10, 1, 10, 1, Qt.AlignVCenter)
        layout.addWidget(backLabel[1], 20, 1, 10, 1)
        layout.addWidget(numberLabel[2], 20, 1, 10, 1, Qt.AlignVCenter)
        layout.addWidget(numberLabel[3], 30, 1, 10, 1, Qt.AlignVCenter)
        
        layout.addWidget(self.pw[0], 0, 2, 10, 2)
        layout.addWidget(self.pw[1], 10, 2, 10, 2)
        layout.addWidget(self.pw[2], 20, 2, 10, 2)
        layout.addWidget(self.pw[3], 30, 2, 10, 2)
        layout.addWidget(self.pbar, 0, 4, 20, 11)
        layout.addWidget(self.pwFFT, 20, 4, 16, 11)
        layout.setColumnStretch(2, 2)
        
        layout.addWidget(self.ChargeLabel[0], 0, 3) 
        layout.addWidget(self.ChargeLabel[1], 10, 3) 
        layout.addWidget(self.ChargeLabel[2], 20, 3) 
        layout.addWidget(self.ChargeLabel[3], 30, 3) 

        layout.addWidget(fftButton[0], 20, 11, 2, 1)
        layout.addWidget(fftButton[1], 20, 12, 2, 1) 
        layout.addWidget(fftButton[2], 20, 13, 2, 1) 
        layout.addWidget(fftButton[3], 20, 14, 2, 1)   
        layout.addWidget(filtersText, 36, 5, 2, 1) 
        layout.addWidget(self.bandstop50, 36, 6, 2, 1) 
        layout.addWidget(self.bandstop60, 36, 7, 2, 1)
        layout.addWidget(self.bandpass, 36, 8, 2, 1) 
        layout.addWidget(self.passLowFreq, 36, 9, 2, 1) 
        layout.addWidget(self.passHighFreq, 36, 10, 2, 1)
        layout.addWidget(plotStyle, 38, 5, 2, 1)
        layout.addWidget(self.signal, 38, 6, 2, 1) 
        layout.addWidget(self.envelope, 38, 7, 2, 1)  
        layout.addWidget(self.envelopeSmoothing, 38, 8, 2, 1)      
        layout.addWidget(self.envelopeSmoothingСoefficient, 38, 9, 2, 1)
        
        vbox.addLayout(layout)
        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)  
        self.showMaximized()
        self.show()
        # Serial monitor
        self.monitor = SerialMonitor(COM, baudRate, self.delay)
        self.monitor.bufferUpdated.connect(self.updateListening, QtCore.Qt.QueuedConnection)
    # Start working
    def start(self):
        self.monitor.running = True
        self.monitor.start()
    # Pause
    def stop(self):
        if self.monitor.running == False:
            self.monitor.running = True
            self.monitor.start()
        else:
            self.monitor.running = False
    # Refresh
    def refresh(self):
        self.l = [0] * 4 #Current point
        self.Time = np.zeros((4, self.dataWidth)) #Tine array
        self.Data = np.zeros((4, self.dataWidth))
        self.DataEnvelope = np.zeros((4, self.dataWidth))
        self.msg_end = bytearray([0])       
        self.loopNumber = 0;
    # Update
    def updateListening(self, msg):
        # Update variables
        self.setWindowTitle("MYOblue GUI v1.0.1 | ELEMYO " + 
                            "    ( " + self.monitor.COM + " , " + str(self.monitor.baudRate) + " baud )")
        s = self.passLowFreq.text()
        if s.isdigit():
            self.passLowFrec = float(s)
        s = self.passHighFreq.text()
        if s.isdigit():
            self.passHighFrec = float(self.passHighFreq.text())
        
        s = self.envelopeSmoothingСoefficient.text()
        try:
            if (float(s) >= 0) and (float(s) <= 1):
                self.MA_alpha= float(s)
        except ValueError:
            pass
        
        
        # Parsing data from serial buffer
        if (len(self.msg_end) > 1):
            msg =  self.msg_end + msg
            self.msg_end = bytearray([0])
        
        if (len(msg) % 246 != 0):
            if(len(msg)>250):
                for i in range(len(msg) - 250, len(msg)-1, 1):
                   if (msg[i] == 0xFF) and (msg[i+1] == 0xFF):
                       self.msg_end = msg[i:]
                       msg = msg[0:i]
                       break
        
        if (len(msg) % 246 == 0):
            for msg_i in range(0, len(msg), 246):
                sensorNum = int(msg[msg_i+2])-1
                MSG_NUM = self.MSG_NUM[sensorNum]
                self.MSG_NUM[sensorNum]= int(msg[msg_i+3] | msg[msg_i+4] << 8 | msg[msg_i+5] << 16)
  
                if (self.l[sensorNum] == 0 or (self.MSG_NUM[sensorNum] < MSG_NUM)):
                        self.Time[sensorNum] = self.Time[self.l.index(max(self.l))]
                        self.l[sensorNum] = self.l[self.l.index(max(self.l))]-246
                        self.Data[sensorNum] = np.zeros(self.dataWidth)
                
                if (self.MSG_NUM[sensorNum] - MSG_NUM > 1 and MSG_NUM != 0):
                    self.Time[sensorNum][self.l[sensorNum] - 1] += self.dt*119*(self.MSG_NUM[sensorNum] - MSG_NUM - 1) 
                
                self.VDD[sensorNum] = round(int(msg[msg_i+6] | msg[msg_i+7] << 8)*0.6/16384*6*2, 2)
                string = "BATTERY CHARGE: " + str(self.VDD[int(msg[msg_i+2])-1])
                while len(string) < 20:
                    string += "0"
                self.ChargeLabel[int(msg[msg_i+2])-1].setText(string)
                
                for i in range(msg_i+8, msg_i+246, 2):
                    if ( self.l[sensorNum] == self.dataWidth):
                        self.l[sensorNum] = 0 
                    self.Data[sensorNum][self.l[sensorNum]]  = (int(msg[i] | msg[i+1] << 8)/16384.0*2.49 - 1.245)*2000
                    if ( self.l[sensorNum] > 0):
                        self.Time[sensorNum][self.l[sensorNum]] = self.Time[sensorNum][self.l[sensorNum] - 1] + self.dt 
                    else:
                        self.Time[sensorNum][self.l[sensorNum]] = self.Time[sensorNum][self.dataWidth - 1] + self.dt
                    self.l[sensorNum] = self.l[sensorNum] + 1
                    self.ms_len[sensorNum] += 1  
                
        # Filtering
        Data = np.zeros((4, self.dataWidth))
        Time = np.zeros((4, self.dataWidth))
        for i in range(4):
            Data[i] = np.concatenate((self.Data[i][self.l[i]: self.dataWidth], self.Data[i][0: self.l[i]]))
            Time[i] = np.concatenate((self.Time[i][self.l[i]: self.dataWidth], self.Time[i][0: self.l[i]]))
        
        self.monitor.delay = self.delay
        if self.bandstop50.isChecked() == 1:
            if self.fs > 110: 
                for i in range(4): Data[i] = self.butter_bandstop_filter(Data[i], 48, 52, self.fs)
            if self.fs > 210: 
                for i in range(4): Data[i] = self.butter_bandstop_filter(Data[i], 98, 102, self.fs)
            if self.fs > 310: 
                for i in range(4): Data[i] = self.butter_bandstop_filter(Data[i], 148, 152, self.fs)
            if self.fs > 410: 
                for i in range(4): Data[i] = self.butter_bandstop_filter(Data[i], 195, 205, self.fs)
            self.monitor.delay = self.delay + 0.03
        if self.bandstop60.isChecked() == 1:
            if self.fs > 130:
                for i in range(4): Data[i] = self.butter_bandstop_filter(Data[i], 58, 62, self.fs)
            if self.fs > 230:
                for i in range(4): Data[i] = self.butter_bandstop_filter(Data[i], 118, 122, self.fs)
            if self.fs > 330:
                for i in range(4): Data[i] = self.butter_bandstop_filter(Data[i], 158, 162, self.fs)
            self.monitor.delay = self.delay + 0.03
        if ((self.bandpass.isChecked() == 1 or (self.signal.isChecked() == 1 and self.envelope.isChecked() == 1)) and self.passLowFrec < self.passHighFrec 
            and self.passLowFrec > 0 and self.fs > 2*self.passHighFrec):
            for i in range(4):
                Data[i] = self.butter_bandpass_filter(Data[i], self.passLowFrec, self.passHighFrec, self.fs)
            self.monitor.delay = self.delay + 0.04
        
        for i in range(4):
            self.DataEnvelope[i][0: self.dataWidth - self.ms_len[i]] = self.DataEnvelope[i][self.ms_len[i]:self.dataWidth]
        for i in range(4):
            for j in range (self.dataWidth - self.ms_len[i], self.dataWidth):
                self.DataEnvelope[i][j] = self.movingAverage(i, Data[i][j], self.MA_alpha)
        self.ms_len = [0]*4
               
        # Shift the boundaries of the graph
        for i in range(4):
            self.pw[i].setXRange(self.timeWidth*(self.Time[i][self.l[i] - 1] // self.timeWidth), self.timeWidth*((self.Time[i][self.l[i] - 1] // self.timeWidth + 1)))            
        
        # Plot raw and envelope data
        if  self.signal.isChecked() == 1 and self.envelope.isChecked() == 1:
            for i in range(4):
                self.p[i].setData(y=Data[i], x=Time[i])
                self.pe[i].setData(y=self.DataEnvelope[i], x=Time[i])
            self.monitor.delay += 0.02
        
        # Plot envelope data            
        if self.signal.isChecked() == 0 and self.envelope.isChecked() == 1:
            for i in range(4):
                self.pe[i].setData(y=self.DataEnvelope[i], x=Time[i])
                self.p[i].clear()
                
        # Plot raw data 
        if self.signal.isChecked() == 1 and self.envelope.isChecked() == 0:
            for i in range(4):
                self.p[i].setData(y=Data[i], x=Time[i])
                self.pe[i].clear()
                        
        # Plot histogram
        for i in range(4):
            self.pb[i].setOpts(height=2*self.DataEnvelope[i][-1])
        
        
        # Plot FFT data
        Y = np.zeros((4, 500))
        for i in range(4):
            Y[i] = abs(fft(Data[i][-501: -1]))/500
            self.FFT[i] = (1-0.85)*Y[i] + 0.85*self.FFT[i]
        X = 1/self.dt*np.linspace(0, 1, 500)
        self.pFFT.setData(y=self.FFT[self.button_group.checkedId() - 1][2: int(len(self.FFT[self.button_group.checkedId() - 1])/2)], x=X[2: int(len(X)/2)]) 
                    
    # Values for butterworth bandpass filter
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a
    # Butterworth bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    # Values for butterworth bandstop filter
    def butter_bandstop(self, lowcut, highcut, fs, order=2):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandstop')
        return b, a
    # Butterworth bandstop filter
    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=4):
        b, a = self.butter_bandstop(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    def movingAverage(self, i, data, alpha):
        wa = 2.0*self.fs*np.tan(3.1416*1/self.fs)
        HPF = (2*self.fs*(data-self.X0[i]) - (wa-2*self.fs)*self.Y0[i])/(2*self.fs+wa)
        self.Y0[i] = HPF
        self.X0[i] = data
        data = HPF
        if data < 0:
            data = -data
        self.MA[i][0] = (1 - alpha)*data + alpha*self.MA[i][0];
        self.MA[i][1] = (1 - alpha)*(self.MA[i][0]) + alpha*self.MA[i][1];
        self.MA[i][2] = (1 - alpha)*(self.MA[i][1]) + alpha*self.MA[i][2];
        return self.MA[i][2]*2
    # Change gain
    def _on_radio_button_clicked(self, button):
        if self.monitor.COM != '':
            self.monitor.ser.write(bytearray([button.Value]))
    # Exit event
    def closeEvent(self, event):
        self.monitor.ser.close()
        event.accept()

# Serial monitor class
class SerialMonitor(QtCore.QThread):
    bufferUpdated = QtCore.pyqtSignal(bytes)
    # Custom constructor
    def __init__(self, COM, baudRate, delay):
        QtCore.QThread.__init__(self)
        self.running = False
        self.filter = False
        self.COM = COM
        self.baudRate = baudRate
        self.baudRate = baudRate
        self.checkPort = 1
        self.delay = delay

    # Listening port
    def run(self):
        while self.running is True:
            while self.COM == '': 
                ports = serial.tools.list_ports.comports(include_links=False)
                for port in ports :
                    self.COM = port.device
                if self.COM != '':
                    time.sleep(0.5)
                    self.ser = serial.Serial(self.COM, self.baudRate)
                    self.checkPort = 0
            while self.checkPort:
                ports = serial.tools.list_ports.comports(include_links=False)
                for port in ports :
                    if self.COM == port.device:
                        time.sleep(0.5)
                        self.ser = serial.Serial(self.COM, self.baudRate)
                        self.checkPort = 0
                   
            # Waiting for data
            while (self.ser.inWaiting() == 0):
                pass
            # Reading data
            msg = self.ser.read( self.ser.inWaiting() )
            if msg:
                #Parsing data
                self.bufferUpdated.emit(msg)
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