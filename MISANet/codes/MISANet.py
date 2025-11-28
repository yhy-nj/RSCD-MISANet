import torch
import torch.nn as nn
import torch.nn.functional as F

import MobileNetV2

# 兼容实现：训练时按样本随机丢弃整条残差支路（stochastic depth）
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 形状：(B,1,1,...)，可同时兼容 [B,C,H,W] / [B,H,W,C] / [B,N,C]
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0:
            rand.div_(keep_prob)
        return x * rand

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class PMFFM(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(PMFFM, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d

        # step1
        self.step1_c1_Conv = nn.Sequential(
            # nn.MaxPool2d(2,2)
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(16, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.step1_c2_Conv = nn.Sequential(
            nn.Conv2d(24, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.step1_c3_Conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.step1_c4_Conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(96, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.step1_c5_Conv = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(320, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.step1_conv = nn.Sequential(
            nn.Conv2d(160, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        # step2
        self.step2_c1_Conv = nn.Sequential(
            # nn.Conv2d(16,64,3,4,1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=False)
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.step2_c2_Conv = nn.Sequential(  # 此处的c2指的是输出的第一张融合特征图[32,64,64]
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.step2_c3_Conv = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.step2_c4_Conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(96, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.step2_c5_Conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(320, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.step2_Conv = nn.Sequential(
            nn.Conv2d(320, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        # step3
        self.step3_c1_Conv = nn.Sequential(
            # nn.Conv2d(16, 128, 3, 8, 2),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=False
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(16, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.step3_c2_Conv = nn.Sequential(  # 此处c2是step1的输出
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(32, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.step3_c3_Conv = nn.Sequential(  # 此处的c3是step2的输出
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.step3_c4_Conv = nn.Sequential(
            nn.Conv2d(96, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.step3_c5_Conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(320, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.step3_Conv = nn.Sequential(
            nn.Conv2d(640, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
        )
        # step4
        self.step4_c1_Conv = nn.Sequential(
            # nn.Conv2d(16, 32, 3, 2, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 128, 3, 2, 1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, 256, 3, 2, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=False)
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(16, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )
        self.step4_c2_Conv = nn.Sequential(  # 此处c2是step1的输出
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(32, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )
        self.step4_c3_Conv = nn.Sequential(  # 此处c3是step2的输出
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )
        self.step4_c4_Conv = nn.Sequential(  #
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )
        self.step4_c5_Conv = nn.Sequential(
            nn.Conv2d(320, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )
        self.step4_Conv = nn.Sequential(
            nn.Conv2d(1280, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )

    def forward(self, c1, c2, c3, c4, c5):
        c1_step1 = self.step1_c1_Conv(c1)
        c2_step1 = self.step1_c2_Conv(c2)
        c3_step1 = self.step1_c3_Conv(c3)
        c4_step1 = self.step1_c4_Conv(c4)
        c5_step1 = self.step1_c5_Conv(c5)

        c_step1 = torch.cat([c1_step1, c2_step1, c3_step1, c4_step1, c5_step1], dim=1)
        s1 = self.step1_conv(c_step1)

        c1_step2 = self.step2_c1_Conv(c1)
        c2_step2 = self.step2_c2_Conv(s1)
        c3_step2 = self.step2_c3_Conv(c3)
        c4_step2 = self.step2_c4_Conv(c4)
        c5_step2 = self.step2_c5_Conv(c5)

        c_step2 = torch.cat([c1_step2, c2_step2, c3_step2, c4_step2, c5_step2], dim=1)
        s2 = self.step2_Conv(c_step2)

        c1_step3 = self.step3_c1_Conv(c1)
        c2_step3 = self.step3_c2_Conv(s1)
        c3_step3 = self.step3_c3_Conv(s2)
        c4_step3 = self.step3_c4_Conv(c4)
        c5_step3 = self.step3_c5_Conv(c5)

        c_step3 = torch.cat([c1_step3, c2_step3, c3_step3, c4_step3, c5_step3], dim=1)
        s3 = self.step3_Conv(c_step3)

        c1_step4 = self.step4_c1_Conv(c1)
        c2_step4 = self.step4_c2_Conv(s1)
        c3_step4 = self.step4_c3_Conv(s2)
        c4_step4 = self.step4_c4_Conv(s3)
        c5_step4 = self.step4_c5_Conv(c5)

        c_step4 = torch.cat([c1_step4, c2_step4, c3_step4, c4_step4, c5_step4], dim=1)
        s4 = self.step4_Conv(c_step4)

        return s1, s2, s3, s4

class AttenShare(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.key1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.key2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.softmax = nn.Softmax(dim=-1)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 4),
            nn.Sigmoid()
        )

        self.gate1=nn.Sequential(
            nn.Conv2d(in_channels*2,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.gate2=nn.Sequential(
            nn.Conv2d(in_channels*2,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )


    def forward(self,input1,input2):
        batch_size, channels, height, width = input1.shape

        diff = torch.abs(input1 - input2)

        g = self.global_att(torch.cat([input1, input2, diff], 1)).view(batch_size, -1, 1 * 1)
        g = g.repeat(1, 1, height * width)

        q1 = self.query1(input1).view(batch_size, -1, height * width).permute(0, 2,
                                                                              1)  # [1, 64, 64, 64]->[1, 8, 64, 64]
        k1 = self.key1(input1).view(batch_size, -1, height * width)  # [1, 64, 64, 64]->[1, 16, 64, 64]->[1, 16, 64*64]
        v1 = self.value1(input1).view(batch_size, -1,
                                      height * width)  # [1, 64, 64, 64]->[1, 64, 64, 64]->[1, 64, 64*64]

        q2 = self.query2(input2).view(batch_size, -1, height * width).permute(0, 2, 1)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        attn_matrix1=torch.bmm(q1,k1)+torch.bmm(q1,g)
        attn_matrix1=self.softmax(attn_matrix1)
        out1=torch.bmm(v1,attn_matrix1.permute(0,2,1)).view(batch_size,channels,height,width)

        attn_matrix2=torch.bmm(q2,k2)+torch.bmm(q2,g)
        attn_matrix2=self.softmax(attn_matrix2)
        out2=torch.bmm(v2,attn_matrix2.permute(0,2,1)).view(batch_size,channels,height,width)

        gate_input1=torch.cat([out1,input1],dim=1)
        gate1=self.gate1(gate_input1)
        out1=gate1*out1+(1-gate1)*input1

        gate_input2=torch.cat([out2,input2],dim=1)
        gate2=self.gate2(gate_input2)
        out2=gate2*out2+(1-gate2)*input2

        return out1,out2

# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)


class Channel_Attention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        avg = self.fc2(self.relu1(self.fc1(avg)))
        max = self.fc2(self.relu1(self.fc1(max)))
        out = avg + max
        out = self.sigmoid(out)
        return out

class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

class CBAM_Attention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(CBAM_Attention, self).__init__()
        self.cha = Channel_Attention(in_ch, reduction=reduction)
        self.spa = Spatial_Attention()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,groups=in_ch,bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.cha(x) * x
        out = self.bn(self.conv(out))+out
        out = self.spa(out) * out
        out = self.relu(out)
        return out

class MSConv3x3(nn.Module):
    def __init__(self,in_ch,dilation=3):
        super().__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,3,1,1,1,1,False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,3,1,3,3,1,False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(2*in_ch,in_ch,1,bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        y1=self.conv1(x)
        y2=self.conv2(x)
        y=torch.cat([y1,y2],dim=1)
        return self.conv3(y)

class My_Attention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(My_Attention, self).__init__()
        self.cbam = CBAM_Attention(in_ch, reduction=reduction)
        self.msconv = MSConv3x3(in_ch,dilation=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(in_ch*2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv1=nn.Sequential(
            nn.Conv2d(2*in_ch,in_ch,1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1_=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.cbam(x)
        out = torch.cat([out, x], dim=1)
        att = self.attention(out)
        out = out * att
        # out = self.conv(out)
        out=self.conv1(out)
        out=self.msconv(out)
        # out=self.conv1(out)
        out_ = out + x
        deepflow=out_
        out = self.conv1_(out_)
        return out,deepflow


class MY_NET(nn.Module):
    def __init__(self, num_classes=2):
        super(MY_NET, self).__init__()

        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2      #64
        # self.swa = NFAM(channles, self.mid_d)
        self.swa = PMFFM(channles, self.mid_d)

        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)

        #以下是特征融合后的上采样
        self.up4=nn.Sequential(
            Double_conv(256,128),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        )
        self.up3=nn.Sequential(
            Double_conv(128,64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up2=nn.Sequential(
            Double_conv(64,32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up1=nn.Sequential(
            Double_conv(224, 112),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.cross1 = AttenShare(32)
        self.cross2 = AttenShare(64)
        self.cross3 = AttenShare(128)
        self.cross4 = AttenShare(256)

        # self.ds1=SupervisionAttentionModule(64)
        # self.ds2=SupervisionAttentionModule(128)
        # self.ds3=SupervisionAttentionModule(256)

        self.ds1=My_Attention(64)
        self.ds2=My_Attention(128)
        self.ds3=My_Attention(256)

        # self.ds1 = SupervisionAttentionModuleV1(64, gated_kernel=7, gated_conv_ratio=1.0, drop_path=0.05)
        # self.ds2 = SupervisionAttentionModuleV1(128, gated_kernel=7, gated_conv_ratio=1.0, drop_path=0.07)
        # self.ds3 = SupervisionAttentionModuleV1(256, gated_kernel=7, gated_conv_ratio=1.0, drop_path=0.1)

        self.output_aux_3=nn.Sequential(
            nn.Conv2d(256,128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(128,num_classes,1)
        )

        self.output_aux_2=nn.Sequential(
            nn.Conv2d(128,64,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(64,num_classes,1)
        )

        self.output_aux_1=nn.Sequential(
            nn.Conv2d(64,32,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(32,num_classes,1)
        )

        self.output=nn.Sequential(
            nn.Conv2d(112,56,1,bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(56,num_classes,1)
        )



    def forward(self, x1, x2):
        h, w = x1.shape[2:]

        # 提取 backbone 特征
        # x_layer0~x_layer4：分别对应不同尺度特征，这里直接用最后一层大尺度特征（或根据需求选其他层）
        # _, _, _, _, x1_feat = self.backbone(x1)  # 取最后一层特征 [B, 320, 8, 8]
        # _, _, _, _, x2_feat = self.backbone(x2)  # 同理

        x1_layer0, x1_layer1, x1_layer2, x1_layer3, x1_layer4 = self.backbone(x1)
        x2_layer0,x2_layer1,x2_layer2,x2_layer3,x2_layer4 = self.backbone(x2)

        x1_layer1, x1_layer2, x1_layer3, x1_layer4 = self.swa(x1_layer0, x1_layer1, x1_layer2, x1_layer3, x1_layer4)
        x2_layer1, x2_layer2, x2_layer3, x2_layer4 = self.swa(x2_layer0, x2_layer1, x2_layer2, x2_layer3, x2_layer4)

        inter1_a, inter1_b = self.cross1(x1_layer1, x2_layer1)  #

        inter2_a, inter2_b = self.cross2(x1_layer2, x2_layer2)  # [4,64,32,32]

        inter3_a, inter3_b = self.cross3(x1_layer3, x2_layer3)  # [4,128,16,16]

        inter4_a, inter4_b = self.cross4(x1_layer4, x2_layer4)  # [4,256,8,8]

        sub_layer1_ = inter1_a - inter1_b
        sub_layer2_ = inter2_a - inter2_b
        sub_layer3_ = inter3_a - inter3_b
        sub_layer4_ = inter4_a - inter4_b

        sp3, aux3 = self.ds3(sub_layer4_)  # [4,256,8,8]
        up4 = self.up4(sp3)  # [4,128,16,16]

        aux_3 = self.output_aux_3(aux3)  # [4,2,8,8]

        add3 = sub_layer3_ + up4  # [4,128,16,16]
        sp2, aux2 = self.ds2(add3)  # [4,128,16,16]
        up3 = self.up3(sp2)  # [4,64,32,32]

        aux_2 = self.output_aux_2(aux2)  # [4,2,16,16]

        add2 = sub_layer2_ + up3  # [4,64,32,32]
        sp1, aux1 = self.ds1(add2)  # [4,64,32,32]
        up2 = self.up2(sp1)  # [4,32,64,64]

        aux_1 = self.output_aux_1(aux1)  # [4,2,32,32]

        add1 = sub_layer1_ + up2  # [4,32,64,64]

        out = torch.cat([F.upsample(add3, add1.shape[2:], mode='bilinear', align_corners=True),
                         F.upsample(add2, add1.shape[2:], mode='bilinear', align_corners=True), add1],dim=1)  # [4,224,64,64]

        out = self.up1(out)  # [4,112,128,128]

        output = self.output(out)  # [4,2,128,128]

        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
        output1 = F.upsample(aux_1, size=(h, w), mode='bilinear', align_corners=True)
        output2 = F.upsample(aux_2, size=(h, w), mode='bilinear', align_corners=True)
        output3 = F.upsample(aux_3, size=(h, w), mode='bilinear', align_corners=True)

        return output, output1, output2, output3

if __name__ == '__main__':
    x1 = torch.rand(4,3,256,256)
    x2 = torch.rand(4,3,256,256)
    model = MY_NET()
    # out, out1, out2, out3 = model(x1,x2)
    # print('out:',out.shape,'out1:',out1.shape,'out2:',out2.shape,'out3:',out3.shape)
    out=model(x1,x2)
    print('Output shape:',out.shape)