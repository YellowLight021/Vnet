import paddle
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import utils
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
import numpy as np
import torchloss
# from monai.losses.dice import DiceLoss

def gen_fake_data():
    fake_output=np.random.rand(1,2,2, 64, 64)
    fake_target=np.random.randint(0,2,(1,2, 64, 64))
    return fake_output,fake_target

if __name__=="__main__":

    output,target=gen_fake_data()
    my_loss=utils.paddle_dice_loss(paddle.to_tensor(output),paddle.to_tensor(target)).item()
    print(my_loss)
    target=torch.as_tensor(target)
    output=torch.as_tensor(output).permute(0,2,3,4,1)
    # torch_loss=DiceLoss()(output.reshape(-1,2),target.view(target.numel()))


    torch_loss=torchloss.dice_error(output.reshape(-1,2),target.view(target.numel()))
    # torch_loss.backward()
    print("my_loss:{},torch_loss:{},abs(my_loss-torch_loss):{}".format(my_loss,torch_loss,abs(my_loss-torch_loss)))
