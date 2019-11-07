# This program is developed by Ali Babolhavaeji
# At 11/7/2019

import compile_ui
from main import Ui_MainWindow
from  PyQt5.QtWidgets import QApplication,QMainWindow
import sys

import matplotlib.pyplot as plt
import gen_pic
import numpy as np

# app=QApplication(sys.argv)
# win=UI.Ui_MainWindow()
# win.show()
# sys.exit(app.exec())

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
            plt.imshow(img)
            # plt.plot(10 ,10, 'go--', linewidth=2, markersize=12)
            self.center_points(plt,annot)
            print(masks.shape)
            fig_masks,ax_masks= plt.subplots(1,masks.shape[2])


            if(masks.shape[2] > 1):
                for i,ax in enumerate(ax_masks):
                    ax.imshow(masks[:,:, i-1])
                    print(i)
            else:
                ax_masks.imshow(masks[: ,:, 0])

            ax=fig_masks.add_subplot(8,1,2)
            #
            ax.imshow(img)

            fig_masks.show()



            # plt.show()
            # print(masks.shape[1])
            # print(annot)

        # plt.subplot(2, 3, 1)
        # plt.imshow(masks[:, :, 0])
        # plt.subplot(2, 3, 2)
        # plt.imshow(masks[:, :, 1])
        # plt.subplot(2, 3, 3)
        # plt.imshow(masks[:, :, 2])
        # plt.show()


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