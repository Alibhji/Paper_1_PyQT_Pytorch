import torch
import gen_pic
import numpy as np
print('The pytorch version:  ',torch.__version__)

from torch.utils.data import  Dataset,dataloader
from torchvision import  transforms, datasets, models

class generated_data(Dataset):
    def __init__(self , height, width, count , transform=None):
        self.input_image , self.target_mask,self.annotation= gen_pic.genrate_random_images(height,width,count)
        # self.input_image=np.reshape(self.input_image,(1,200,200))
        self.transform= transform

    def __len__(self):
            return (len(self.input_image))

    def __getitem__(self,idx):
        image = self.input_image[idx]
        mask= self.target_mask[idx]
        if self.transform:
            image = self.transform(image)
        return [image,mask]



if __name__ == '__main__':
    anno=generated_data(200,200,10)
    print(anno.annotation)
    print(anno.input_image.shape)
    # x,y,a= zip(*[gen_pic.generate_img_and_mask(200,200,num=1,circle=True) for i in range(5)])
    # x,y,a=gen_pic.genrate_random_images(200,200,40)

    # print((list(a)))
    # X = np.asarray(x) * 255
    #
    # X = X.repeat(3, axis=1).transpose(0,2,3,1).astype(np.uint8)
    # print(x.shape)
    # Y=np.asarray(y)


