import torch
import torch.nn as nn
import gen_pic
import time
import copy
import numpy as np
print('The pytorch version:  ',torch.__version__)

from torch.utils.data import  Dataset,dataloader
from torchvision import  transforms, datasets, models

class generated_data(Dataset):
    def __init__(self ,config ,str__='train'):


        F_ = config['objecs_Fill']
        C_, S_, T_ = config['objecs_circle'], config['objecs_squre'], config['objecs_triangle']
        n_objs=config['objecs_num']

        if(str__=='train'):
            N_=config['dataset_size_train']
        else:
            N_ = config['dataset_size_test']


        self.input_image , self.target_mask,self.annotation= gen_pic.genrate_random_images(config['image_size_H'],
                                                                                           config['image_size_W'],
                                                                                           N_,circle=C_,squre=S_,triangle=T_,fill=F_
                                                                                           ,num_each_obj=n_objs)

        # self.input_image=np.reshape(self.input_image,(1,200,200))
        # print('::::::::::::::',(list(self.input_image)[0].shape))
        print('::::::::::::::', (list(self.target_mask)[0].shape))
        self.transform= transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
            return (len(self.input_image))

    def __getitem__(self,idx):
        image = self.input_image[idx]
        mask= self.target_mask[idx]
        image = self.transform(image)

        return [image,mask]



if __name__ == '__main__':
    pass
    # anno=generated_data(200,200,10)
    # print(anno.annotation)
    # print(anno.input_image.shape)
    # x,y,a= zip(*[gen_pic.generate_img_and_mask(200,200,num=1,circle=True) for i in range(5)])
    # x,y,a=gen_pic.genrate_random_images(200,200,40)

    # print((list(a)))
    # X = np.asarray(x) * 255
    #
    # X = X.repeat(3, axis=1).transpose(0,2,3,1).astype(np.uint8)
    # print(x.shape)
    # Y=np.asarray(y)




def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


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
                    outputs = model(inputs)
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