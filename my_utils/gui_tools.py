from  PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtGui import QColor



class utils():
    def __init__(self,ui):
        ui.ui.text_Logger.setText("The program is started.")
        self.color()
        self.ui=ui

    def color(self):

        self.black = QColor(0, 0, 0)
        self.red = QColor(255, 0, 0)
        self.green = QColor(0, 100, 0)


    def logging(self,message=" ", type='info'):
        if(type=='info'):
            self.ui.ui.text_Logger.setTextColor(self.green)
            self.ui.ui.text_Logger.append(message)
        if(type=='red'):
            self.ui.ui.text_Logger.setTextColor(self.red)
            self.ui.ui.text_Logger.append(message)
            self.ui.ui.text_Logger.setTextColor(self.black)


