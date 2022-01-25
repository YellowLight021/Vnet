import paddle
import paddle.nn as nn
import paddle.nn.functional as F



def _make_nConV(outchans,nConvs):
    layers=[]
    for _ in range(nConvs):
        layers.append(Block(outchans))
    return nn.Sequential(*layers)

class InputTransition(nn.Layer):
    def __init__(self,outchans):
        super(InputTransition, self).__init__()
        self.conv1=nn.Conv3D(1,outchans,kernel_size=5,padding=2)
        self.bn1=nn.BatchNorm3D(outchans)
        self.relu1=nn.PReLU()
    def forward(self,x):
        out=self.relu1(self.bn1(self.conv1(x)))
        x16=paddle.concat([x,x,x,x,x,x,x,x,
                           x,x,x,x,x,x,x,x],1)
        return paddle.add(x16,out)
class Block(nn.Layer):
    def __init__(self,nchans):
        super(Block,self).__init__()
        self.conv=nn.Conv3D(nchans,nchans,kernel_size=5,padding=2)
        self.bn=nn.BatchNorm3D(nchans)
        self.relu=nn.PReLU()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))
class DownTransition(nn.Layer):
    def __init__(self,inchans,nConvs):
        super(DownTransition,self).__init__()
        outchans=2*inchans
        self.down_conv=nn.Conv3D(inchans,outchans,kernel_size=2,stride=2)
        self.bn1=nn.BatchNorm3D(outchans)
        self.relu1=nn.PReLU()
        # self.relu2=nn.PReLU()

        self.ops=_make_nConV(outchans,nConvs)

    def forward(self,x):
        down=self.relu1(self.bn1(self.down_conv(x)))
        out=self.ops(down)
        return paddle.add(down,out)


class UpTransition(nn.Layer):
    def __init__(self,inchans,outchans,nConvs):
        super(UpTransition,self).__init__()
        self.outchans=outchans
        self.inchans=inchans
        #如果输出维度和输入维度相等，但是后面有一个相加操作
        self.up_conv=nn.Conv3DTranspose(inchans,outchans//2,kernel_size=2,stride=2)
        self.bn1 = nn.BatchNorm3D(outchans//2)
        self.relu1=nn.PReLU()
        self.ops=_make_nConV(outchans,nConvs)

    def forward(self,x,skipx):
        out=self.relu1(self.bn1(self.up_conv(x)))
        x=paddle.concat([out,skipx],axis=1)
        x=self.ops(x)
        # if self.outchans==self.inchans:
        out=paddle.concat([out,out],axis=1)
        return x+out

class OutputTransition(nn.Layer):
    def __init__(self,inchans,nll):
        super(OutputTransition, self).__init__()
        self.conv1=nn.Conv3D(inchans,2,kernel_size=1)
        self.bn1=nn.BatchNorm3D(2)
        self.relu=nn.PReLU()
        if nll:
            self.softmax=F.log_softmax
        else:
            self.softmax=F.softmax
    def forward(self, x):
        out=self.relu(self.bn1(self.conv1(x)))
        return out,self.softmax(out,axis=1)



class VNet(nn.Layer):

    def __init__(self,nll=False):
        super(VNet,self).__init__()
        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 2)
        self.down_tr64 = DownTransition(32, 3)
        self.down_tr128 = DownTransition(64, 3)
        self.down_tr256 = DownTransition(128, 3)
        self.up_tr256= UpTransition(256, 256, 2)
        self.up_tr128 = UpTransition(256, 128, 2)
        self.up_tr64 = UpTransition(128, 64, 1)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.out_tr = OutputTransition(32,nll)
        # self.out_dict={}

    def forward(self,x):
        out16=self.in_tr(x)
        # self.out_dict['out16']=out16
        out32=self.down_tr32(out16)
        # self.out_dict['out32']=out32
        out64= self.down_tr64(out32)
        # self.out_dict['out64']=out64
        out128 = self.down_tr128(out64)
        # self.out_dict['128']=out128
        out256 = self.down_tr256(out128)
        # self.out_dict['256']=out256
        out=self.up_tr256(out256,out128)
        # self.out_dict['out1']=out
        out = self.up_tr128(out, out64)
        # self.out_dict['out2']=out
        out = self.up_tr64(out, out32)
        # self.out_dict['out3']=out
        out = self.up_tr32(out, out16)
        # self.out_dict['out4']=out
        out05,out = self.out_tr(out)
        # self.out_dict['out05']=out05
        return out

if __name__=="__main__":


    model=VNet(True)
    # model=model.cuda()
    model.eval()
    for i in range(100):
        test_input = paddle.randn([1, 1, 64, 128, 128]).cuda()
        out_put=model(test_input)
        print(out_put.shape)
