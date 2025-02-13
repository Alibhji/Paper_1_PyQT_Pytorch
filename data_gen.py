# This program is developed by Ali Babolhavaeji
# At 11/7/2019

import compile_ui
from main import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gen_pic
import numpy as np

import learning
# import torch

from my_utils import gui_tools

import torch
from torch.utils.data import  DataLoader
import torchvision.utils
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler


class AppWindow(QMainWindow):
    def __init__(self):
        super(AppWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.conf_ui()

    def conf_ui(self):
        self.ui.btn_gen.clicked.connect(self.pic_gen)
        self.ui.btn_gendataset.clicked.connect(self.gen_dataset)
        self.ui.btn_train.clicked.connect(self.train)
        self.tools = gui_tools.utils(self)
        self.config = {}
        self.update_config()
        self.config.update({'image_counter': 0})

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tools.logging("The model is set.")
        self.tools.logging("The {} device is selected.".format(device))
        self.model = learning.AliNet()
        self.model = self.model.to(device)
        self.dataloaders=self.dataloaders.to(device)
        summary(self.model, input_size=(3, 25, 25))
        self.optimizer_ft = optim.Adam(self.model.parameters(), lr=1e-4)
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer_ft, step_size=25, gamma=0.1)
        self.model = learning.train_model(self.model,self.dataloaders, self.optimizer_ft, self.exp_lr_scheduler, num_epochs=10)



    def gen_dataset(self):
        self.update_config()

        dataset_train = learning.generated_data(self.config , 'train')
        dataset_test = learning.generated_data(self.config , 'test')
        # dataset_train = learning.generated_data(self.config,'train')
        # dataset_test =  learning.generated_data(self.config,'test')

        # print(len(self.dataset_train) ,len(self.dataset_test))

        self.datasets = {
            'train': dataset_train, 'test': dataset_test
        }
        # print(len(self.datasets['train']))
        batch_size = 50
        self.dataloaders = {
            'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0),
            'test': DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
        }

        dataset_sizes = {
            x: len(self.datasets[x]) for x in self.datasets.keys()
        }
        # print(dataset_sizes)

        self.tools.logging(
            "Datasets are generated \nTrain_data size:{}\nTest_data size:{}".format(dataset_sizes['train'],dataset_sizes['test']))

        def reverse_transform(inp):
            inp = inp.numpy().transpose((1, 2, 0))
            inp = np.clip(inp, 0, 1)
            inp = (inp * 255).astype(np.uint8)

            return inp

        # Get a batch of training data
        inputs, masks = next(iter(self.dataloaders['train']))

        # print(inputs.shape, masks.shape)
        for x in [inputs.numpy(), masks.numpy()]:
            # print(x.min(), x.max(), x.mean(), x.std())
            self.tools.logging("[Train Statistics and Assosiated Masks]: \nMin={} \nMax={} \nMean={} \nSTD={}".format(x.min(), x.max(), x.mean(), x.std()),'red')

        # plt.imshow(reverse_transform(inputs[3]))
        # plt.show()

    def center_points(self, plt, anntation):
        # objes={}
        # objes=anntation
        # objes.values()
        for obj in anntation:
            print(anntation[obj][1])
            plt.plot(anntation[obj][1], anntation[obj][0], 'ro--', linewidth=1, markersize=2)

    def pic_gen(self):

        self.update_config()

        H_, W_, N_ = self.config['image_size_H'], self.config['image_size_W'], self.config['objecs_num']
        F_ = self.config['objecs_Fill']
        C_, S_, T_ = self.config['objecs_circle'], self.config['objecs_squre'], self.config['objecs_triangle']

        if C_ or S_ or T_:

            img, masks, annot = gen_pic.generate_img_and_mask(H_, W_, num_each_obj=N_, fill=F_, circle=C_, squre=S_,
                                                              triangle=T_)
            masks = masks.transpose(1, 2, 0)
            # img = img.transpose(2, 0, 1)
            img = np.reshape(img, (H_, W_))

            self.config['image_counter'] += 1
            self.tools.logging("Generated image:" + str(self.config['image_counter']))
            # print("Genrated_image", self.config['image_counter'])

            # print(img.shape)
            fig = plt.figure(figsize=(9, 15))
            gs = gridspec.GridSpec(nrows=2, ncols=masks.shape[2])
            title = ["Circle Mask", "Squre Mask", "Tria angle Mask"]
            #
            for i in range(masks.shape[2]):
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(masks[:, :, i])
                ax.set_title(title[i])
            #
            ax = fig.add_subplot(gs[1, :])
            self.center_points(ax, annot)
            ax.imshow(img)
            ax.set_title("Generated Image")
            # # plt.tight_layout()
            plt.show()

    def update_config(self):
        self.config.update({'image_size_H': int(self.ui.in_img_size_H.text())})
        self.config.update({'image_size_W': int(self.ui.in_img_size_W.text())})
        self.config.update({'objecs_num': int(self.ui.in_num_objs.text())})
        self.config.update({'objecs_Fill': int(self.ui.in_objs_fill.text())})
        self.config.update({'objecs_circle': (self.ui.chk_box_circle.isChecked())})
        self.config.update({'objecs_squre': (self.ui.chk_box_squre.isChecked())})
        self.config.update({'objecs_triangle': (self.ui.chk_box_triangle.isChecked())})
        self.config.update({'dataset_size_test': int(self.ui.btn_test_size.text())})
        self.config.update({'dataset_size_train': int(self.ui.btn_train_size.text())})
        if self.ui.chbox_show_configuration.isChecked():
            self.tools.logging("[Configuration]: \n" + (str(self.config)), "red")
        # print(self.config)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec())
