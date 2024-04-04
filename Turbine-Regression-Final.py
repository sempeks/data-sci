import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QListWidget, QMainWindow, QRadioButton, QDialog, QVBoxLayout, QLineEdit
from PyQt5.QtWidgets import QSizePolicy
from PyQt5 import QtCore

class GraphDialog(QDialog):
    def __init__(self, parent=None):
        super(GraphDialog, self).__init__(parent)

        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setWindowTitle("Graph Window")

    def plot_graph(self, x, y, y_hat, title, startFrom, endAt):
        self.ax.clear()
        self.ax.plot(x, y, "--b", label="y")
        self.ax.plot(x, y_hat, "-g", label="y_hat")
        self.ax.legend(["y", "y_hat"])
        self.ax.set_xlim(startFrom,endAt)
        self.ax.set_ylim(0,1)
        self.ax.set_title(title)
        self.canvas.draw()
        
class Regresyoner(QMainWindow):
    
    def __init__(self):
        
        super().__init__()
        
        self.df = pd.DataFrame()
        
        self.current_directory = os.getcwd()
        
        self.N = 0

        self.Result = np.array(1)
        self.x = np.array(1)
        self.y = np.array(1)
        self.z = np.array(1)
        self.q = np.array(1)
        self.t = np.array(1)
        self.w = np.array(1)
        
        self.MSE1 = 0
        self.MSE2 = 0
        self.MSE3 = 0
        
        self.ResultHat1 = np.array(1)
        self.ResultHat2 = np.array(1)
        self.ResultHat3 = np.array(1)
        
        self.name = " "

        self.setWindowTitle("WindTurbine Regressioner v1.0")
        self.setGeometry(100, 100, 1000, 500)
        self.setMinimumSize(1000, 500)
        self.setMaximumSize(1000, 500)

        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        
        self.label = QLabel('VERİ REGRESYON MERKEZİ', self.centralwidget)
        self.label.setGeometry(10, 0, 200, 30)

        self.label_2 = QLabel('Bu veri ile hangi işlemi yapmak istiyorsun?', self.centralwidget)
        self.label_2.setGeometry(10, 30, 400, 30)

        self.label_6 = QLabel('Hızlı veri seçmek ister misin?', self.centralwidget)
        self.label_6.setGeometry(10, 140, 200, 20)

        self.file_label = QLabel('Dosya Seçimi:', self.centralwidget)
        self.file_label.setGeometry(10, 105, 200, 20)

        self.file_button = QPushButton('Dosya Seç', self.centralwidget)
        self.file_button.setGeometry(100, 100, 100, 30)
        self.file_button.clicked.connect(self.open_file_dialog)

        self.gradient_button = QPushButton('Gradient Regression Yap', self.centralwidget)
        self.gradient_button.setGeometry(10, 60, 180, 30)
        self.gradient_button.clicked.connect(self.run_gradient_descent)

        self.LSE_button = QPushButton('LSE Regression Yap', self.centralwidget)
        self.LSE_button.setGeometry(200, 60, 180, 30)
        self.LSE_button.clicked.connect(self.LSERegression)
        
        self.startFrom = QLineEdit("0", self) 
        self.startFrom.setGeometry(390, 60, 100, 30) 

        self.startLabel = QLabel("Başlangıç", self) 
        self.startLabel.setGeometry(390, 40, 100, 20)
        
        self.startLabel = QLabel("Grafik Sınırları:", self) 
        self.startLabel.setGeometry(390, 20, 100, 20)
        
        self.endAt = QLineEdit("1000", self) 
        self.endAt.setGeometry(500, 60, 100, 30) 

        self.endLabel = QLabel("Bitiş", self) 
        self.endLabel.setGeometry(500, 40, 100, 20)

        self.radio_button_1 = QRadioButton('Lokasyon 1', self.centralwidget)
        self.radio_button_1.setGeometry(10, 160, 120, 20)
        self.radio_button_1.clicked.connect(self.LoadDataLocation1)

        self.radio_button_2 = QRadioButton('Lokasyon 2', self.centralwidget)
        self.radio_button_2.setGeometry(150, 160, 120, 20)
        self.radio_button_2.clicked.connect(self.LoadDataLocation2)

        self.radio_button_3 = QRadioButton('Lokasyon 3', self.centralwidget)
        self.radio_button_3.setGeometry(290, 160, 120, 20)
        self.radio_button_3.clicked.connect(self.LoadDataLocation3)
        
        self.radio_button_3 = QRadioButton('Lokasyon 4', self.centralwidget)
        self.radio_button_3.setGeometry(430, 160, 120, 20)
        self.radio_button_3.clicked.connect(self.LoadDataLocation4)

        self.head4data_label = QLabel('Verinin İlk 5 Satırı', self.centralwidget)
        self.head4data_label.setGeometry(10, 200, 200, 20)

        self.head4data = QListWidget(self.centralwidget)
        self.head4data.setGeometry(10, 220, 980, 98)

        self.tail4data_label = QLabel('Verinin Son 5 Satırı', self.centralwidget)
        self.tail4data_label.setGeometry(10, 320, 200, 20)

        self.tail4data = QListWidget(self.centralwidget)
        self.tail4data.setGeometry(10, 340, 980, 98)

        self.progress_label = QLabel('', self.centralwidget)
        self.progress_label.setGeometry(10, 500, 800, 20)

        self.footer_label = QLabel('WindTurbine Regressioner by 53rd Team', self.centralwidget)
        self.footer_label.setGeometry(800, 480, 200, 20)  
        
        self.head4data.setWordWrap(True)
        self.head4data.setTextElideMode(QtCore.Qt.ElideNone)
        self.tail4data.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.tail4data.setWordWrap(True)
        self.tail4data.setTextElideMode(QtCore.Qt.ElideNone)
        self.tail4data.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def plot_graph(self):
        
        graph_dialog_1 = GraphDialog(self)
        graph_dialog_2 = GraphDialog(self)
        graph_dialog_3 = GraphDialog(self)

        graph_dialog_1.plot_graph(range(len(self.Result)), self.Result, self.ResultHat1, self.name + " Linear Modelling with MSE: " + 
                                  str(self.MSE1)[2:-2], int(self.startFrom.text()),int(self.endAt.text()))
        graph_dialog_2.plot_graph(range(len(self.Result)), self.Result, self.ResultHat2, self.name + " Polynomial Modelling with MSE: " + 
                                  str(self.MSE2)[2:-2], int(self.startFrom.text()),int(self.endAt.text()))
        graph_dialog_3.plot_graph(range(len(self.Result)), self.Result, self.ResultHat3, self.name + " Mixed Modelling with MSE: " + 
                                  str(self.MSE3)[2:-2], int(self.startFrom.text()),int(self.endAt.text()))

        graph_dialog_1.exec_()
        graph_dialog_2.exec_()
        graph_dialog_3.exec_()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Lütfen bir .CSV dosyası seçiniz!", "", "CSV Files (*.csv)", options=options)
        if file_name:
            try:
                
                self.head4data.clear()
                self.tail4data.clear()
                
                self.df = pd.read_csv(file_name)
                
                self.x = np.array(self.df.iloc[:, 1]).reshape(-1,1)
                self.y = np.array(self.df.iloc[:, 2]).reshape(-1,1)
                self.w = np.array(self.df.iloc[:, 3]).reshape(-1,1)
                self.z = np.array(self.df.iloc[:, 4]).reshape(-1,1)
                self.q = np.array(self.df.iloc[:, 6]).reshape(-1,1)
                self.t = np.array(self.df.iloc[:, 8]).reshape(-1,1)
                self.Result = np.array(self.df.iloc[:, -1]).reshape(-1,1)
                
                self.N = len(self.df)
                
                self.name = file_name

                head = '\n'.join([str(self.df.head()) for index in range (5)])
                tail = '\n'.join([str(self.df.tail()) for index in range (5)])

                self.head4data.addItem(head)
                self.tail4data.addItem(tail)
            except Exception as e:
                print(f"Error: {e}")
    
    def LoadDataLocation1(self):
        try:
            
            self.head4data.clear()
            self.tail4data.clear()
            
            self.df = pd.read_csv(self.current_directory + "\Location1.csv")
            
            self.x = np.array(self.df.iloc[:, 1]).reshape(-1,1)
            self.y = np.array(self.df.iloc[:, 2]).reshape(-1,1)
            self.w = np.array(self.df.iloc[:, 3]).reshape(-1,1)
            self.z = np.array(self.df.iloc[:, 4]).reshape(-1,1)
            self.q = np.array(self.df.iloc[:, 6]).reshape(-1,1)
            self.t = np.array(self.df.iloc[:, 8]).reshape(-1,1)
            self.Result = np.array(self.df.iloc[:, -1]).reshape(-1,1)

            self.N = len(self.df)
            
            self.name = "Location1"

            head = '\n'.join([str(self.df.head()) for index in range (5)])
            tail = '\n'.join([str(self.df.tail()) for index in range (5)])

            self.head4data.addItem(head)
            self.tail4data.addItem(tail)
        except Exception as e:
            print(f"Error: {e}")
            
    def LoadDataLocation2(self):
        try:
            
            self.head4data.clear()
            self.tail4data.clear()

            self.df = pd.read_csv(self.current_directory + "\Location2.csv")
            
            self.x = np.array(self.df.iloc[:, 1]).reshape(-1,1)
            self.y = np.array(self.df.iloc[:, 2]).reshape(-1,1)
            self.w = np.array(self.df.iloc[:, 3]).reshape(-1,1)
            self.z = np.array(self.df.iloc[:, 4]).reshape(-1,1)
            self.q = np.array(self.df.iloc[:, 6]).reshape(-1,1)
            self.t = np.array(self.df.iloc[:, 8]).reshape(-1,1)
            self.Result = np.array(self.df.iloc[:, -1]).reshape(-1,1)
            
            self.N = len(self.df)
            
            self.name = "Location2"

            head = '\n'.join([str(self.df.head()) for index in range (5)])
            tail = '\n'.join([str(self.df.tail()) for index in range (5)])

            self.head4data.addItem(head)
            self.tail4data.addItem(tail)
        except Exception as e:
            print(f"Error: {e}")
            
    def LoadDataLocation3(self):
        try:
            
            self.head4data.clear()
            self.tail4data.clear()
            
            self.df = pd.read_csv(self.current_directory + "\Location3.csv")
            
            self.x = np.array(self.df.iloc[:, 1]).reshape(-1,1)
            self.y = np.array(self.df.iloc[:, 2]).reshape(-1,1)
            self.w = np.array(self.df.iloc[:, 3]).reshape(-1,1)
            self.z = np.array(self.df.iloc[:, 4]).reshape(-1,1)
            self.q = np.array(self.df.iloc[:, 6]).reshape(-1,1)
            self.t = np.array(self.df.iloc[:, 8]).reshape(-1,1)
            self.Result = np.array(self.df.iloc[:, -1]).reshape(-1,1)
            
            self.N = len(self.df)
            
            self.name = "Location3"

            head = '\n'.join([str(self.df.head()) for index in range (5)])
            tail = '\n'.join([str(self.df.tail()) for index in range (5)])

            self.head4data.addItem(head)
            self.tail4data.addItem(tail)
        except Exception as e:
            print(f"Error: {e}")
            
            
    def LoadDataLocation4(self):
        try:
            
            self.head4data.clear()
            self.tail4data.clear()
            
            self.df = pd.read_csv(self.current_directory + "\Location4.csv")
            
            self.x = np.array(self.df.iloc[:, 1]).reshape(-1,1)
            self.y = np.array(self.df.iloc[:, 2]).reshape(-1,1)
            self.w = np.array(self.df.iloc[:, 3]).reshape(-1,1)
            self.z = np.array(self.df.iloc[:, 4]).reshape(-1,1)
            self.q = np.array(self.df.iloc[:, 6]).reshape(-1,1)
            self.t = np.array(self.df.iloc[:, 8]).reshape(-1,1)
            self.Result = np.array(self.df.iloc[:, -1]).reshape(-1,1)
            
            self.N = len(self.df)
            
            self.name = "Location4"

            head = '\n'.join([str(self.df.head()) for index in range (5)])
            tail = '\n'.join([str(self.df.tail()) for index in range (5)])

            self.head4data.addItem(head)
            self.tail4data.addItem(tail)

        except Exception as e:
            print(f"Error: {e}")
            
                
    
    def LSERegression(self):
        
        Fi1 = np.array(np.zeros((self.df.shape[0], 7)))
        Fi2 = np.array(np.zeros((self.df.shape[0], 13)))
        Fi3 = np.array(np.zeros((self.df.shape[0], 25)))
        
        print(str(self.df.shape))
        print(self.df.shape[1])

        for i in range (self.N):
            Fi1[i,:] = [1, 
                        self.x[i][0], 
                        self.y[i][0], 
                        self.z[i][0], 
                        self.q[i][0], 
                        self.t[i][0], 
                        self.w[i][0]]
            
            Fi2[i,:] = [1, 
                        self.x[i][0], self.x[i][0]*self.x[i][0], 
                        self.y[i][0], self.y[i][0]*self.y[i][0], 
                        self.z[i][0], self.z[i][0]*self.z[i][0], 
                        self.q[i][0], self.q[i][0]*self.q[i][0], 
                        self.t[i][0], self.t[i][0]*self.t[i][0], 
                        self.w[i][0], self.w[i][0]*self.w[i][0]]
            
            Fi3[i,:] = [1, 
                        self.x[i][0], np.cos(np.deg2rad(self.x[i][0])), np.exp(np.dot(self.x[i][0], -1)), np.sqrt(abs(self.x[i][0])), 
                        self.y[i][0], np.cos(np.deg2rad(self.y[i][0])), np.exp(np.dot(self.y[i][0], -1)), np.sqrt(abs(self.y[i][0])), 
                        self.z[i][0], np.cos(np.deg2rad(self.z[i][0])), np.exp(np.dot(self.z[i][0], -1)), np.sqrt(abs(self.z[i][0])), 
                        self.q[i][0], np.cos(np.deg2rad(self.q[i][0])), np.exp(np.dot(self.q[i][0], -1)), np.sqrt(abs(self.q[i][0])), 
                        self.t[i][0], np.cos(np.deg2rad(self.t[i][0])), np.exp(np.dot(self.t[i][0], -1)), np.sqrt(abs(self.t[i][0])), 
                        self.w[i][0], np.cos(np.deg2rad(self.w[i][0])), np.exp(np.dot(self.w[i][0], -1)), np.sqrt(abs(self.w[i][0]))]
                
        A1 = Fi1.T @ Fi1
        B1 = Fi1.T @ self.Result
        
        A2 = Fi2.T @ Fi2
        B2 = Fi2.T @ self.Result
        
        A3 = Fi3.T @ Fi3
        B3 = Fi3.T @ self.Result
        
        teta1 = np.linalg.inv(A1) @ B1
        teta2 = np.linalg.inv(A2) @ B2
        teta3 = np.linalg.inv(A3) @ B3
        
        self.ResultHat1 = Fi1 @ teta1
        self.ResultHat2 = Fi2 @ teta2
        self.ResultHat3 = Fi3 @ teta3
        
        e1 = self.Result - self.ResultHat1
        e2 = self.Result - self.ResultHat2
        e3 = self.Result - self.ResultHat3
        
        self.MSE1 = e1.T @ e1 / (self.N)
        self.MSE2 = e2.T @ e2 / (self.N)
        self.MSE3 = e3.T @ e3 / (self.N)
        
        self.plot_graph()
        
        
    def gradient_descent(self, Fi, learning_rate=0.018, num_iterations=1000, batch_size=32):
        theta = np.zeros((Fi.shape[1], 1))

        for _ in range(num_iterations):
            indices = np.random.permutation(Fi.shape[0])

            for start in range(0, Fi.shape[0], batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_Fi = Fi[batch_indices]
                batch_Result = self.Result[batch_indices]

                predictions = batch_Fi @ theta
                errors = predictions - batch_Result
                gradients = batch_Fi.T @ errors / batch_size

                theta = theta - learning_rate * gradients

        return theta

    def run_gradient_descent(self):
        Fi = np.column_stack([np.ones(len(self.Result)), self.x, self.y, self.z, self.q, self.t, self.w])

        scaler = StandardScaler()
        
        Fi_scaled = scaler.fit_transform(Fi)

        teta = self.gradient_descent(Fi_scaled)

        self.ResultHat1 = Fi_scaled @ teta - np.min(teta)
        e1 = self.Result - self.ResultHat1
        self.MSE1 = e1.T @ e1 / self.N

        graph_dialog = GraphDialog(self)

        graph_dialog.plot_graph(range(self.N), self.Result, self.ResultHat1, self.name + " Gradient Descent Linear Modelling with MSE: " + 
                                str(self.MSE1)[2:-2], int(self.startFrom.text()),int(self.endAt.text()))
        
        graph_dialog.exec_() 
        

def main():
    app = QApplication(sys.argv)
    window = Regresyoner()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.expand_frame_repr', True)
    main()