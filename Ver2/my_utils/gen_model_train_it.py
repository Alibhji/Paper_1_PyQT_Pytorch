import torch
import torch.nn as nn

from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from gen_pic import genrate_random_images
import numpy as np

import time
import copy


class Custom_Model_():
    def __init__(self, config):
        self.config = config
        print("Custom model is created.")
        self.dataset()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.tools.logging("The model is set.")
        # self.tools.logging("The {} device is selected.".format(device))
        self.model = AliNet()
        self.model = self.model.to(device)
        summary(self.model, input_size=(3, 25, 25))
        self.optimizer_ft = optim.Adam(self.model.parameters(), lr=1e-4)
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer_ft, step_size=25, gamma=0.1)
        self.model = train_model(self.model,self.dataloaders, self.optimizer_ft, self.exp_lr_scheduler, num_epochs=10)

    def dataset(self):
        dataset_train = generated_data(self.config, str__='train')
        dataset_test = generated_data(self.config, str__='test')

        self.datasets = {
            'train': dataset_train, 'test': dataset_test
        }

        batch_size = 50
        self.dataloaders = {
            'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0),
            'test': DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
        }

        dataset_sizes = {
            x: len(self.datasets[x]) for x in self.datasets.keys()
        }
        # print(dataset_sizes)

        # self.tools.logging(
        #     "Datasets are generated \nTrain_data size:{}\nTest_data size:{}".format(dataset_sizes['train'],
        #                                                                             dataset_sizes['test']))

        def reverse_transform(inp):
            inp = inp.numpy().transpose((1, 2, 0))
            inp = np.clip(inp, 0, 1)
            inp = (inp * 255).astype(np.uint8)

            return inp

        # Get a batch of training data
        inputs, masks = next(iter(self.dataloaders['train']))

        # print(inputs.shape, masks.shape)
        # for x in [inputs.numpy(), masks.numpy()]:
        #     # print(x.min(), x.max(), x.mean(), x.std())
        #     self.tools.logging(
        #         "[Train Statistics and Assosiated Masks]: \nMin={} \nMax={} \nMean={} \nSTD={}".format(x.min(), x.max(),
        #                                                                                                x.mean(),
        #                                                                                                x.std()), 'red')

        # plt.imshow(reverse_transform(inputs[3]))
        # plt.show()

class AliNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 6, 3, padding=1)
        self.out = nn.Conv2d(6, 1, 3, padding=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.out(x)

        return x

class generated_data(Dataset):
    def __init__(self, config, str__='train'):

        F_ = config['objecs_Fill']
        C_, S_, T_ = config['objecs_circle'], config['objecs_squre'], config['objecs_triangle']
        n_objs = config['objecs_num']

        if (str__ == 'train'):
            N_ = config['dataset_size_train']
        else:
            N_ = config['dataset_size_test']

        self.input_image, self.target_mask, self.annotation = genrate_random_images(config['image_size_H'],
                                                                                    config['image_size_W'],
                                                                                    N_, circle=C_, squre=S_,
                                                                                    triangle=T_, fill=F_
                                                                                    , num_each_obj=n_objs)

        # self.input_image=np.reshape(self.input_image,(1,200,200))
        # print('::::::::::::::',(list(self.input_image)[0].shape))
        print('::::::::::::::', (list(self.target_mask)[0].shape))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return (len(self.input_image))

    def __getitem__(self, idx):
        image = self.input_image[idx]
        mask = self.target_mask[idx]
        image = self.transform(image)

        return [image, mask]


from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model,dataloaders, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # model=model.to(device)
                    # outputs = model(inputs).to(device)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model