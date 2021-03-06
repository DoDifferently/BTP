
from PyQt5 import QtCore, QtGui, QtWidgets
import base, mfcc, svm, gmm, decision_tree, NN

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(666, 428)
        MainWindow.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_3.addWidget(self.lineEdit)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setStyleSheet("font: italic 14pt \"Calibri\";\n"
"selection-background-color: rgb(170, 255, 255);\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(0, 0, 0);")
        self.pushButton.setAutoDefault(False)
        self.pushButton.setDefault(False)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_3.addWidget(self.pushButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setStyleSheet("\n"
"font: 75 italic 24pt \"Calibri\";\n"
"background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 235, 235, 206), stop:0.35 rgba(255, 188, 188, 80), stop:0.4 rgba(255, 162, 162, 80), stop:0.425 rgba(255, 132, 132, 156), stop:0.44 rgba(252, 128, 128, 80), stop:1 rgba(255, 255, 255, 0));")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setStyleSheet("font: 75 14pt \"Calibri\";")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setStyleSheet("font: 75 14pt \"Calibri\";")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setStyleSheet("font: 75 14pt \"Calibri\";")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setStyleSheet("font: 75 14pt \"Calibri\";")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.dt = QtWidgets.QLabel(self.centralwidget)
        self.dt.setStyleSheet("font: italic 12pt \"Calibri\";")
        self.dt.setText("")
        self.dt.setAlignment(QtCore.Qt.AlignCenter)
        self.dt.setObjectName("dt")
        self.horizontalLayout_2.addWidget(self.dt)
        self.gmm = QtWidgets.QLabel(self.centralwidget)
        self.gmm.setStyleSheet("font: italic 12pt \"Calibri\";")
        self.gmm.setText("")
        self.gmm.setAlignment(QtCore.Qt.AlignCenter)
        self.gmm.setObjectName("gmm")
        self.horizontalLayout_2.addWidget(self.gmm)
        self.svm = QtWidgets.QLabel(self.centralwidget)
        self.svm.setStyleSheet("font: italic 12pt \"Calibri\";")
        self.svm.setText("")
        self.svm.setAlignment(QtCore.Qt.AlignCenter)
        self.svm.setObjectName("svm")
        self.horizontalLayout_2.addWidget(self.svm)
        self.nn = QtWidgets.QLabel(self.centralwidget)
        self.nn.setStyleSheet("font: italic 12pt \"Calibri\";")
        self.nn.setText("")
        self.nn.setAlignment(QtCore.Qt.AlignCenter)
        self.nn.setObjectName("nn")
        self.horizontalLayout_2.addWidget(self.nn)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setStyleSheet("font: 80 14pt \"Calibri\";")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_4.addWidget(self.label_10)
        self.ar = QtWidgets.QLabel(self.centralwidget)
        self.ar.setStyleSheet("font: italic 12pt \"Calibri\";")
        self.ar.setText("")
        self.ar.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.ar.setObjectName("ar")
        self.horizontalLayout_4.addWidget(self.ar)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)

#        self.train()
        self.pushButton.clicked.connect(self.showDialog)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def showDialog(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, 'Open file', '/home')
        print fname[0].split('/')[-2]
        self.ar.setText(fname[0].split('/')[-2])
        self.lineEdit.setText(fname[0])
        dt = decision_tree.test(fname[0])
        self.dt.setText(dt)
        gmm1 = gmm.test(fname[0])
        self.gmm.setText(gmm1)
        svm1 = svm.test(fname[0])
        self.svm.setText(svm1)
        nn = NN.test(fname[0])
        self.nn.setText(nn)

    def train(self):
        decision_tree.train()
        gmm.train()
        svm.train()
        NN.train()
        
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Select File"))
        self.label.setText(_translate("MainWindow", "Our Predictions"))
        self.label_2.setText(_translate("MainWindow", "Decision Tree"))
        self.label_3.setText(_translate("MainWindow", "GMM"))
        self.label_4.setText(_translate("MainWindow", "SVM"))
        self.label_5.setText(_translate("MainWindow", "NN"))
        self.label_10.setText(_translate("MainWindow", "Actual Result :"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())