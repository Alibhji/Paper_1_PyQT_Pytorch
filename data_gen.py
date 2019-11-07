# This program is developed by Ali Babolhavaeji
# At 11/7/2019

import compile_ui
from main import Ui_MainWindow
from  PyQt5.QtWidgets import QApplication,QMainWindow
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gen_pic
import numpy as np

class AppWindow(QMainWindow):
    def __init__(self):
        super(AppWindow, self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.conf_ui()

    def conf_ui(self):
        self.ui.btn_gen.clicked.connect(self.pic_gen)
        self.config={}
        self.update_config()
        self.config.update({'image_counter': 0})

    def center_points(self,plt, anntation):
        # objes={}
        # objes=anntation
        # objes.values()
        for obj in anntation:
            print(anntation[obj][1])
            plt.plot(anntation[obj][1] ,anntation[obj][0], 'ro--', linewidth=1, markersize=2)

    def pic_gen(self):

        self.update_config()
        H_,W_,N_ =self.config['image_size_H'] ,self.config['image_size_W'],self.config['objecs_num']
        F_=self.config['objecs_Fill']
        C_ , S_ , T_ = self.config['objecs_circle'] ,self.config['objecs_squre'],self.config['objecs_triangle']

        if C_ or S_ or T_:
            img, masks ,annot= gen_pic.generate_img_and_mask(H_, W_, N_, F_, circle=C_, squre=S_, triangle=T_)
            masks = masks.transpose(1, 2, 0)
            self.config['image_counter'] += 1
            print("Genrated_image", self.config['image_counter'])

            print(masks.shape)
            fig = plt.figure(figsize=(9, 15))
            gs = gridspec.GridSpec(nrows=2, ncols=masks.shape[2])
            title=["Circle Mask","Squre Mask","Tria angle Mask"]
            #
            for i in range (masks.shape[2]):
                ax= fig.add_subplot(gs[0, i])
                ax.imshow(masks[:, :, i])
                ax.set_title(title[i])

            ax = fig.add_subplot(gs[1, :])
            self.center_points(ax, annot)
            ax.imshow(img)
            ax.set_title("Generated Image")
            # plt.tight_layout()
            plt.show()





    def update_config(self):
        self.config.update({'image_size_H': int(self.ui.in_img_size_H.text())})
        self.config.update({'image_size_W': int(self.ui.in_img_size_W.text())})
        self.config.update({'objecs_num': int(self.ui.in_num_objs.text())})
        self.config.update({'objecs_Fill': int(self.ui.in_objs_fill.text())})
        self.config.update({'objecs_circle': (self.ui.chk_box_circle.isChecked())})
        self.config.update({'objecs_squre': (self.ui.chk_box_squre.isChecked())})
        self.config.update({'objecs_triangle': (self.ui.chk_box_triangle.isChecked())})
        # print(self.config)


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=AppWindow()
    window.show()
    sys.exit(app.exec())