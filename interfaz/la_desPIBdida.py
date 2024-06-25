from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QFileDialog, QPushButton, QDialog, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.image as mpimg

from functions import *


class PopupDialog(QDialog):
        def __init__(self, message):
                super().__init__()
                #self.setWindowTitle("Popup Dialog")
                self.file_path = ''
                
                file_dialog = QFileDialog()
                file_dialog.setNameFilters(["Image files (*.jpg *.jpeg *.png *.gif *.bmp *.tiff)"])
                file_dialog.selectNameFilter("Image files (*.jpg *.jpeg *.png *.gif *.bmp *.tiff)")

                if file_dialog.exec_():
                        self.file_path = file_dialog.selectedFiles()[0]
                else:
                        self.file_path = None

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.subplots_adjust(0,0,1,1)
        super(MplCanvas, self).__init__(fig)


class Ui_MainWindow(object):
        def setupUi(self, MainWindow):
                MainWindow.setObjectName("MainWindow")
                MainWindow.resize(660, 655)
                MainWindow.setStyleSheet("background-color: rgb(200, 200, 200);")
                self.centralwidget = QtWidgets.QWidget(MainWindow)
                self.centralwidget.setObjectName("centralwidget")
                self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
                self.verticalLayout_3.setObjectName("verticalLayout_3")
                self.verticalLayout_main = QtWidgets.QVBoxLayout()
                self.verticalLayout_main.setContentsMargins(-1, 0, -1, 0)
                self.verticalLayout_main.setSpacing(10)
                self.verticalLayout_main.setObjectName("verticalLayout_main")
                self.horizontalLayout_img = QtWidgets.QHBoxLayout()
                self.horizontalLayout_img.setSpacing(6)
                self.horizontalLayout_img.setObjectName("horizontalLayout_img")
                self.widget_imgOrig = QtWidgets.QWidget(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.widget_imgOrig.sizePolicy().hasHeightForWidth())
                self.widget_imgOrig.setSizePolicy(sizePolicy)
                self.widget_imgOrig.setMinimumSize(QtCore.QSize(400, 0))
                self.widget_imgOrig.setStyleSheet("border: 2px solid rgb(0,0,0);")
                self.widget_imgOrig.setObjectName("widget_imgOrig")
                self.horizontalLayout_img.addWidget(self.widget_imgOrig)
                self.verticalLayout_imgSecund = QtWidgets.QVBoxLayout()
                self.verticalLayout_imgSecund.setSpacing(5)
                self.verticalLayout_imgSecund.setObjectName("verticalLayout_imgSecund")
                self.widget_imgKmeans = QtWidgets.QWidget(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.widget_imgKmeans.sizePolicy().hasHeightForWidth())
                self.widget_imgKmeans.setSizePolicy(sizePolicy)
                self.widget_imgKmeans.setMinimumSize(QtCore.QSize(180, 130))
                self.widget_imgKmeans.setStyleSheet("border: 2px solid rgb(0,0,0);")
                self.widget_imgKmeans.setObjectName("widget_imgKmeans")
                self.verticalLayout_imgSecund.addWidget(self.widget_imgKmeans)
                self.widget_imgVasos = QtWidgets.QWidget(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.widget_imgVasos.sizePolicy().hasHeightForWidth())
                self.widget_imgVasos.setSizePolicy(sizePolicy)
                self.widget_imgVasos.setMinimumSize(QtCore.QSize(180, 130))
                self.widget_imgVasos.setStyleSheet("border: 2px solid rgb(0,0,0);")
                self.widget_imgVasos.setObjectName("widget_imgVasos")
                self.verticalLayout_imgSecund.addWidget(self.widget_imgVasos)
                self.widget_imgRegresion = QtWidgets.QWidget(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.widget_imgRegresion.sizePolicy().hasHeightForWidth())
                self.widget_imgRegresion.setSizePolicy(sizePolicy)
                self.widget_imgRegresion.setMinimumSize(QtCore.QSize(180, 130))
                self.widget_imgRegresion.setStyleSheet("border: 2px solid rgb(0,0,0);")
                self.widget_imgRegresion.setObjectName("widget_imgRegresion")
                self.verticalLayout_imgSecund.addWidget(self.widget_imgRegresion)
                self.horizontalLayout_img.addLayout(self.verticalLayout_imgSecund)
                self.horizontalLayout_img.setStretch(0, 3)
                self.horizontalLayout_img.setStretch(1, 1)
                self.verticalLayout_main.addLayout(self.horizontalLayout_img)
                self.horizontalLayout_aux = QtWidgets.QHBoxLayout()
                self.horizontalLayout_aux.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
                self.horizontalLayout_aux.setContentsMargins(0, -1, 0, -1)
                self.horizontalLayout_aux.setSpacing(2)
                self.horizontalLayout_aux.setObjectName("horizontalLayout_aux")
                spacerItem = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
                self.horizontalLayout_aux.addItem(spacerItem)
                self.verticalLayout = QtWidgets.QVBoxLayout()
                self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
                self.verticalLayout.setSpacing(1)
                self.verticalLayout.setObjectName("verticalLayout")
                self.button_select_img = QtWidgets.QPushButton(self.centralwidget)
                self.button_select_img.setMinimumSize(QtCore.QSize(200, 45))
                self.button_select_img.setMaximumSize(QtCore.QSize(200, 45))
                font = QtGui.QFont()
                font.setPointSize(20)
                self.button_select_img.setFont(font)
                self.button_select_img.setStyleSheet("background-color: rgb(220, 220, 220);")
                self.button_select_img.setObjectName("button_select_img")
                self.verticalLayout.addWidget(self.button_select_img)
                self.button_start_diagnosis = QtWidgets.QPushButton(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.button_start_diagnosis.sizePolicy().hasHeightForWidth())
                self.button_start_diagnosis.setSizePolicy(sizePolicy)
                self.button_start_diagnosis.setMinimumSize(QtCore.QSize(250, 70))
                self.button_start_diagnosis.setMaximumSize(QtCore.QSize(250, 70))
                font = QtGui.QFont()
                font.setPointSize(22)
                self.button_start_diagnosis.setFont(font)
                self.button_start_diagnosis.setStyleSheet("background-color: rgb(230, 230, 230);")
                self.button_start_diagnosis.setObjectName("button_start_diagnosis")
                self.button_start_diagnosis.setEnabled(False)
                self.verticalLayout.addWidget(self.button_start_diagnosis)
                self.horizontalLayout_aux.addLayout(self.verticalLayout)
                spacerItem1 = QtWidgets.QSpacerItem(140, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
                self.horizontalLayout_aux.addItem(spacerItem1)
                self.verticalLayout_diagostico = QtWidgets.QVBoxLayout()
                self.verticalLayout_diagostico.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
                self.verticalLayout_diagostico.setSpacing(0)
                self.verticalLayout_diagostico.setObjectName("verticalLayout_diagostico")
                self.horizontalLayout = QtWidgets.QHBoxLayout()
                self.horizontalLayout.setContentsMargins(-1, 0, -1, -1)
                self.horizontalLayout.setSpacing(0)
                self.horizontalLayout.setObjectName("horizontalLayout")
                self.diagnostico_txt = QtWidgets.QLabel(self.centralwidget)
                self.diagnostico_txt.setEnabled(True)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.diagnostico_txt.sizePolicy().hasHeightForWidth())
                self.diagnostico_txt.setSizePolicy(sizePolicy)
                self.diagnostico_txt.setMinimumSize(QtCore.QSize(220, 50))
                self.diagnostico_txt.setMaximumSize(QtCore.QSize(16777215, 66))
                font = QtGui.QFont()
                font.setPointSize(20)
                self.diagnostico_txt.setFont(font)
                self.diagnostico_txt.setStyleSheet("border: 1px solid rgb(0,0,0);\n"
        "border-right: 0px;\n"
        "")
                self.diagnostico_txt.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
                self.diagnostico_txt.setObjectName("diagnostico_txt")
                self.horizontalLayout.addWidget(self.diagnostico_txt)
                self.diagnostico = QtWidgets.QLabel(self.centralwidget)
                self.diagnostico.setMinimumSize(QtCore.QSize(200, 50))
                font = QtGui.QFont()
                font.setPointSize(20)
                self.diagnostico.setFont(font)
                self.diagnostico.setStyleSheet("border: 1px solid rgb(0,0,0);\n"
        "border-left: 0px;\n"
        "")
                self.diagnostico.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
                self.diagnostico.setObjectName("diagnostico")
                self.horizontalLayout.addWidget(self.diagnostico)
                self.verticalLayout_diagostico.addLayout(self.horizontalLayout)
                self.horizontalLayout_vasos = QtWidgets.QHBoxLayout()
                self.horizontalLayout_vasos.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
                self.horizontalLayout_vasos.setSpacing(0)
                self.horizontalLayout_vasos.setObjectName("horizontalLayout_vasos")
                self.vasos_txt = QtWidgets.QLabel(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.vasos_txt.sizePolicy().hasHeightForWidth())
                self.vasos_txt.setSizePolicy(sizePolicy)
                self.vasos_txt.setMinimumSize(QtCore.QSize(220, 50))
                self.vasos_txt.setMaximumSize(QtCore.QSize(16777215, 66))
                font = QtGui.QFont()
                font.setPointSize(18)
                self.vasos_txt.setFont(font)
                self.vasos_txt.setStyleSheet("border: 1px solid rgb(0,0,0);\n"
        "border-right: 0px;\n"
        "border-top: 0px;")
                self.vasos_txt.setObjectName("vasos_txt")
                self.horizontalLayout_vasos.addWidget(self.vasos_txt)
                self.n_vasos = QtWidgets.QLabel(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.n_vasos.sizePolicy().hasHeightForWidth())
                self.n_vasos.setSizePolicy(sizePolicy)
                self.n_vasos.setMinimumSize(QtCore.QSize(200, 50))
                self.n_vasos.setMaximumSize(QtCore.QSize(16777215, 66))
                font = QtGui.QFont()
                font.setPointSize(18)
                self.n_vasos.setFont(font)
                self.n_vasos.setStyleSheet("border: 1px solid rgb(0,0,0);\n"
        "border-left: 0px;\n"
        "border-top: 0px;")
                self.n_vasos.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
                self.n_vasos.setObjectName("n_vasos")
                self.horizontalLayout_vasos.addWidget(self.n_vasos)
                self.verticalLayout_diagostico.addLayout(self.horizontalLayout_vasos)
                self.horizontalLayout_r2 = QtWidgets.QHBoxLayout()
                self.horizontalLayout_r2.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
                self.horizontalLayout_r2.setSpacing(0)
                self.horizontalLayout_r2.setObjectName("horizontalLayout_r2")
                self.r2_txt = QtWidgets.QLabel(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.r2_txt.sizePolicy().hasHeightForWidth())
                self.r2_txt.setSizePolicy(sizePolicy)
                self.r2_txt.setMinimumSize(QtCore.QSize(220, 50))
                self.r2_txt.setMaximumSize(QtCore.QSize(16777215, 66))
                font = QtGui.QFont()
                font.setPointSize(18)
                self.r2_txt.setFont(font)
                self.r2_txt.setStyleSheet("border: 1px solid rgb(0,0,0);\n"
        "border-right: 0px;\n"
        "border-top: 0px;")
                self.r2_txt.setObjectName("r2_txt")
                self.horizontalLayout_r2.addWidget(self.r2_txt)
                self.r2 = QtWidgets.QLabel(self.centralwidget)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self.r2.sizePolicy().hasHeightForWidth())
                self.r2.setSizePolicy(sizePolicy)
                self.r2.setMinimumSize(QtCore.QSize(200, 50))
                self.r2.setMaximumSize(QtCore.QSize(16777215, 66))
                font = QtGui.QFont()
                font.setPointSize(18)
                self.r2.setFont(font)
                self.r2.setStyleSheet("border: 1px solid rgb(0,0,0);\n"
        "border-left: 0px;\n"
        "border-top: 0px;")
                self.r2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
                self.r2.setObjectName("r2")
                self.horizontalLayout_r2.addWidget(self.r2)
                self.verticalLayout_diagostico.addLayout(self.horizontalLayout_r2)
                self.horizontalLayout_aux.addLayout(self.verticalLayout_diagostico)
                spacerItem2 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
                self.horizontalLayout_aux.addItem(spacerItem2)
                self.horizontalLayout_aux.setStretch(0, 1)
                self.horizontalLayout_aux.setStretch(2, 3)
                self.horizontalLayout_aux.setStretch(3, 2)
                self.horizontalLayout_aux.setStretch(4, 1)
                self.verticalLayout_main.addLayout(self.horizontalLayout_aux)
                self.verticalLayout_main.setStretch(0, 5)
                self.verticalLayout_main.setStretch(1, 1)
                self.verticalLayout_3.addLayout(self.verticalLayout_main)
                MainWindow.setCentralWidget(self.centralwidget)
                self.menubar = QtWidgets.QMenuBar(MainWindow)
                self.menubar.setGeometry(QtCore.QRect(0, 0, 660, 24))
                self.menubar.setObjectName("menubar")
                MainWindow.setMenuBar(self.menubar)
                self.statusbar = QtWidgets.QStatusBar(MainWindow)
                self.statusbar.setObjectName("statusbar")
                MainWindow.setStatusBar(self.statusbar)

                self.retranslateUi(MainWindow)
                QtCore.QMetaObject.connectSlotsByName(MainWindow)


                #Seleccionar y Mostrar imagen original
                self.layout_imgOrig = QVBoxLayout(self.widget_imgOrig)

                self.canvas1 = MplCanvas(self, width=6, height=5, dpi=120)
                self.canvas1.axes.axis('off')
                self.layout_imgOrig.addWidget(self.canvas1)
                self.button_select_img.clicked.connect(self.on_click_select_button)

                #Mostrar imagen K-means
                self.layout_imgKmeans = QVBoxLayout(self.widget_imgKmeans)
                self.canvas2 = MplCanvas(self, width=6, height=5, dpi=120)
                self.canvas2.axes.axis('off')
                self.layout_imgKmeans.addWidget(self.canvas2)

                #Mostrar imagen vasos
                self.layout_imgVasos = QVBoxLayout(self.widget_imgVasos)
                self.canvas3 = MplCanvas(self, width=6, height=5, dpi=120)
                self.canvas3.axes.axis('off')
                self.layout_imgVasos.addWidget(self.canvas3)

                #Mostrar imagen regresion
                self.layout_imgRegresion = QVBoxLayout(self.widget_imgRegresion)
                self.canvas4 = MplCanvas(self, width=6, height=5, dpi=120)
                self.canvas4.axes.axis('off')
                self.layout_imgRegresion.addWidget(self.canvas4)

                #On click diagnostico
                self.button_start_diagnosis.clicked.connect(self.on_click_diagnosis_button)


        def retranslateUi(self, MainWindow):
                _translate = QtCore.QCoreApplication.translate
                MainWindow.setWindowTitle(_translate("MainWindow", "La desPIBdida"))
                self.button_select_img.setText(_translate("MainWindow", "Seleccionar Imagen"))
                self.button_start_diagnosis.setText(_translate("MainWindow", "Realizar Diagnóstico"))
                self.diagnostico_txt.setText(_translate("MainWindow", "Diagnóstico"))
                self.diagnostico.setText(_translate("MainWindow", "-"))
                self.vasos_txt.setText(_translate("MainWindow", "N˙ de vasos/infiltraciones"))
                self.n_vasos.setText(_translate("MainWindow", "-"))
                self.r2_txt.setText(_translate("MainWindow", "<html><head/><body><p>R<span style=\" vertical-align:super;\">2 </span></p></body></html>"))
                self.r2.setText(_translate("MainWindow", "-"))
        
        def plot_img(self, img, canvas, title, font = 7, eti = False):
                canvas.axes.clear()
                if eti:
                        canvas.axes.imshow(img, cmap='nipy_spectral')   
                else:
                        canvas.axes.imshow(img, cmap="gray", vmin = 0, vmax = 255)
                canvas.axes.axis('off')
                canvas.axes.set_title(title)
                canvas.axes.title.set_size(font)
                canvas.draw()

        def clear_plot(self, canvas):
                canvas.axes.clear()
                canvas.axes.imshow([[255,255],[255,255]], cmap="gray", vmin = 0, vmax = 255)
                canvas.axes.axis('off')
                canvas.draw()
                
        def on_click_select_button(self):
                self.popup = PopupDialog("")
                if self.popup.file_path:
                        self.img_orig_path = self.popup.file_path
                        
                        self.img_orig = cv2.imread(self.img_orig_path,0)

                        self.img_preprocess = pre_process(self.img_orig)

                        self.plot_img(self.img_preprocess, self.canvas1, 'Original Pre-procesada', font = 12)

                        self.button_start_diagnosis.setEnabled(True)

                else:
                        self.clear_plot(self.canvas1)
                        self.clear_plot(self.canvas2)
                        self.clear_plot(self.canvas3)
                        self.clear_plot(self.canvas4)

                        self.n_vasos.setText('-')
                        self.r2.setText('-')
                        self.diagnostico.setText('-')

                        self.button_start_diagnosis.setEnabled(False)

        def on_click_diagnosis_button(self):
                self.img_Kmeans, self.img_vessels, self.vessels_count = vessels(self.img_preprocess)

                self.img_line, self.r2_value = best_fit_corr(self.img_Kmeans, mode = 'parabolic')

                self.final_diagnosis = classify(self.vessels_count, self.r2_value)

                self.plot_img(self.img_Kmeans, self.canvas2, 'K-means')
                self.plot_img(self.img_vessels, self.canvas3, 'Vasos Etiquetados', eti = True)
                self.plot_img(self.img_line, self.canvas4, 'Linea de Ajuste Morfologica')

                self.n_vasos.setText(str(self.vessels_count))
                self.r2.setText(f'{np.round(self.r2_value,3)}')
                self.diagnostico.setText(self.final_diagnosis)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
