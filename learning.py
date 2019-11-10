import torch
import gen_pic
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


