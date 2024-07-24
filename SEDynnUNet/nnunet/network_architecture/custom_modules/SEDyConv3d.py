


import torch
import torch.nn as nn
import torch.nn.functional as F

class SEDyConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, dyAttBlocks, reduction=0.0625, kernel_num=5, stride=1, dilation=1, bias=None, kernel_size=[3,3,3], padding=[1,1,1], groups=1,
                  spatial_convert_channels=64):
        super(SEDyConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.dyAttBlocks = dyAttBlocks
        self.temperature = 1.0
        self.dyAttBlocks_flg = False
        self.threshold = torch.nn.Threshold(0.4, 0.0)
        if dyAttBlocks is not None:
            self.dyAttBlocks_flg = True

        self.attention = Attention(in_channels, out_channels, kernel_size, groups=groups,
                                       reduction=reduction, kernel_num=kernel_num, stride=stride, padding=padding,
                                       spatial_convert_channels=spatial_convert_channels, dyAttBlocks=dyAttBlocks)

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2]),
                                   requires_grad=True)
        self.leakyReLU = nn.LeakyReLU(inplace=False)

        self.out_ins = []
        for i in range(kernel_num):
            self.out_ins.append(nn.InstanceNorm3d(out_channels, affine=True))
        self.out_ins = nn.ModuleList(self.out_ins)

        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight[0], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)
        self.temperature = temperature
        print(self.temperature)

    def update_dyAttBlocks(self, dyAttBlocks):
        if self.dyAttBlocks_flg:
            self.dyAttBlocks = self.threshold(torch.sigmoid(dyAttBlocks/self.temperature))


    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention = self.attention(x)
        batch_size, in_channels, height, width, depth = x.size()
        if self.dyAttBlocks_flg:
            channel_attention = channel_attention*(self.dyAttBlocks[:,1].reshape(-1,1,1,1,1)[0:x.size(0),:])+1
            filter_attention = filter_attention*(self.dyAttBlocks[:,0].reshape(-1,1,1,1,1)[0:x.size(0),:])+1
        else:
            print("++ dyAttBlocks_flg False, x.shape", x.shape)

        x = x * channel_attention
        x = x.reshape(1, -1, height, width, depth)
        kernelOutSpatial_attention = spatial_attention[0].reshape(batch_size, self.kernel_num, -1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])#out
        spatial_attention = spatial_attention[1].reshape(batch_size, self.kernel_num, 1, -1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])#in


        if self.dyAttBlocks_flg:
            kernelOutSpatial_attention = kernelOutSpatial_attention*(self.dyAttBlocks[:,2].reshape(-1, 1,1,1,1)[0:x.size(0),:])+1
            spatial_attention = spatial_attention*(self.dyAttBlocks[:,3].reshape(-1, 1,1,1,1)[0:x.size(0),:])+1
            print("++ kernelOutSpatial_attention dyAttBlocks", self.dyAttBlocks)

        aggregate_weight = spatial_attention * kernelOutSpatial_attention * self.weight.unsqueeze(dim=0)
        output = None

        for i in range(self.kernel_num):
            att_weight = aggregate_weight[:,i,:].reshape(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
            outputx = F.conv3d(x, weight=att_weight, bias=None, stride=self.stride, padding=self.padding,
                             dilation=self.dilation, groups=self.groups * batch_size)
            outputx = outputx.view(batch_size, self.out_channels, outputx.size(-3), outputx.size(-2), outputx.size(-1))
            outputx = self.out_ins[i](outputx)
            outputx = self.leakyReLU(outputx)
            if i==0:
                output = outputx
            else:
                output += outputx
        output = output * filter_attention
        return output


    def forward(self, x):
        return self._forward_impl(x)

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dyAttBlocks, groups=1, reduction=0.0625, kernel_num=5, spatial_convert_channels=64, min_channel=32, stride=1, padding=[1,1,1]):
        super(Attention, self).__init__()
        attention_channel = max(int(in_channels * reduction), min_channel)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_channels, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm3d(attention_channel)
        self.relu = nn.LeakyReLU(inplace=False)
        self.spatial_convert_in = nn.InstanceNorm3d(attention_channel, affine=True)
        if in_channels <= spatial_convert_channels:
            self.spatial_convert = nn.Conv3d(in_channels, attention_channel, (3, 3, 3), bias=False, stride=(2, 2, 2), padding=1)# for 3D
        else:
            self.spatial_convert = nn.Conv3d(in_channels, attention_channel, (3, 3, 3), bias=False, stride=(1, 1, 1), padding=1)# for 3D
        if in_channels != out_channels:
            self.align_cn = nn.Conv3d(out_channels, in_channels, 1, bias=False)
            self.align_in = nn.InstanceNorm3d(in_channels, affine=True)
        self.spatial_fc_conK = nn.Conv3d(attention_channel, out_channels, 1, bias=False, stride=1, padding=0)
        self.spatial_fc_conK_in = nn.InstanceNorm3d(out_channels, affine=True)
        self.spatial_fc_conQ = nn.Conv3d(attention_channel, 2*kernel_size[0]*kernel_size[1]*kernel_size[2]*self.kernel_num, 1, bias=False, stride=1, padding=0)
        self.spatial_fc_conQ_in = nn.InstanceNorm3d(2*kernel_size[0]*kernel_size[1]*kernel_size[2]*self.kernel_num, affine=True)
        self.spatial_fc_conKernel_in = nn.InstanceNorm3d(2*out_channels*self.kernel_num, affine=True)

        self.func_spatial = self.get_att_kernel_spatial_attention
        self.channel_fc = nn.Conv3d(attention_channel, in_channels, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_channels == groups and in_channels == out_channels:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv3d(attention_channel, out_channels, 1, bias=True)
            self.func_filter = self.get_filter_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = self.channel_fc(x).view(x.size(0), -1, 1, 1, 1)
        return torch.sigmoid(channel_attention/ self.temperature)

    def get_filter_attention(self, x):
        filter_attention =self.filter_fc(x).view(x.size(0), -1, 1, 1, 1)
        return  torch.sigmoid(filter_attention / self.temperature)

    # kernel spatial attention
    def get_att_kernel_spatial_attention(self, in_x):
        xK = self.spatial_fc_conK(in_x)
        xK = self.spatial_fc_conK_in(xK)
        xQ = self.spatial_fc_conQ(in_x)
        xQ = self.spatial_fc_conQ_in(xQ)
        xQ = xQ.reshape(in_x.size(0), xQ.size(1), -1)
        xK = xK.reshape(in_x.size(0), -1, xK.size(1))
        dx = in_x.size(2)*in_x.size(3)
        xQ = torch.matmul(xQ, xK)/(dx)
        xQ = self.spatial_fc_conKernel_in(xQ.reshape(in_x.size(0), -1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))

        kernelOutSpatial_attention = xQ[:, 0:xQ.size(1) // 2, :]
        Spatial_attention = xQ[:, xQ.size(1) // 2:, :]
        if self.in_channels != self.out_channels:
            Spatial_attention = self.align_cn(Spatial_attention)
            Spatial_attention = self.align_in(Spatial_attention)

        return torch.sigmoid(kernelOutSpatial_attention / self.temperature), torch.sigmoid(Spatial_attention / self.temperature)

    def forward(self, x):

        in_x = self.spatial_convert(x)
        in_x = self.spatial_convert_in(in_x)
        in_x = self.relu(in_x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(in_x)