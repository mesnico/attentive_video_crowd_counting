from einops import rearrange
from fastai.layers import ConvLayer, NormType
from fastai.torch_imports import *
from torchvision import models

from crossattention_augmented_conv import AugmentedConv
from variables import BE_CHANNELS, MODE


class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    def __make_weight(self, feature, scale_feature):
        weight_feature = feature - scale_feature
        return torch.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                        self.scales]
        weights = [self.__make_weight(feats, scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0] * weights[0] + multi_scales[1] * weights[1] + multi_scales[2] * weights[
            2] + multi_scales[3] * weights[3]) / (weights[0] + weights[1] + weights[2] + weights[3])] + [feats]

        bottle = self.bottleneck(torch.cat(overall_features, 1))

        return self.relu(bottle)


class FeatureFusionModel(nn.Module):
    def __init__(self, mode, img_feat_dim, txt_feat_dim, common_space_dim):
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            pass
        elif mode == 'weighted':
            self.alphas = nn.Sequential(
                nn.Linear(img_feat_dim + txt_feat_dim, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 2))
            self.img_proj = nn.Linear(img_feat_dim, common_space_dim)
            self.txt_proj = nn.Linear(txt_feat_dim, common_space_dim)
            self.post_process = nn.Sequential(
                nn.Linear(common_space_dim, common_space_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                # nn.Linear(common_space_dim, common_space_dim)
            )

    def forward(self, img_feat, txt_feat):
        if self.mode == 'concat':
            out_feat = torch.cat((img_feat, txt_feat), 1)
            return out_feat
        elif self.mode == 'weighted':
            b, c, h, w = img_feat.shape
            img_feat = rearrange(img_feat, 'b c h w -> (b h w) c')
            txt_feat = rearrange(txt_feat, 'b c h w -> (b h w) c')
            concat_feat = torch.cat((img_feat, txt_feat), dim=1)
            alphas = torch.sigmoid(self.alphas(concat_feat))  # B x 2
            img_feat_norm = img_feat
            txt_feat_norm = txt_feat
            out_feat = img_feat_norm * alphas[:, 0].unsqueeze(1) + txt_feat_norm * alphas[:, 1].unsqueeze(1)
            out_feat = self.post_process(out_feat)
            out_feat = rearrange(out_feat, '(b h w) c -> b c h w', b=b, h=h, w=w)

            return out_feat


class BMM(nn.Module):
    def __init__(self):
        super(BMM, self).__init__()

    def forward(self, q, k):
        return torch.bmm(q, k)


class SelfAttention(nn.Module):
    " Self attention Layer"

    def __init__(self, in_dim, activation='relu'):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.bmm = BMM()
        self.softmax = nn.Softmax(dim=1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = self.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out


class CANNet2s(nn.Module):
    def __init__(self, load_weights=False, uncertainty=False):
        super(CANNet2s, self).__init__()
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=BE_CHANNELS, batch_norm=True, dilation=True)
        self.output_layer = nn.Conv2d(64, 20 if uncertainty else 10, kernel_size=1)
        self.relu = nn.ReLU()

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self, x_prev, x):
        x_prev = self.frontend(x_prev)
        x = self.frontend(x)

        x_prev = self.context(x_prev)
        x = self.context(x)

        x = torch.cat((x_prev, x), 1)

        x = self.backend(x)

        x = self.output_layer(x)
        x = self.relu(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# class XAtt(nn.Module):
#     def __init__(self, n_channels):
#         self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels,n_channels,n_channels)]
#         self.gamma = nn.Parameter(torch.FloatTensor([0.]))
#
#     def _conv(self,n_in,n_out):
#         return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)
#
#     def forward(self, x, y):
#         #Notation from the paper.
#         size = x.size()
#         x = x.view(*size[:2],-1)
#         f1, g1, h1 = self.query(x),self.key(y),self.value(x)    # first XA
#         beta1 = F.softmax(torch.bmm(f1.transpose(1,2), g1), dim=1)
#         ox = self.gamma * torch.bmm(h1, beta1) + x
#
#         f2, g2, h2 = self.query(y), self.key(x), self.value(y)  # second XA
#         beta2 = F.softmax(torch.bmm(f2.transpose(1, 2), g2), dim=1)
#         oy = self.gamma * torch.bmm(h2, beta2) + y
#
#         out = ox + oy
#
#         return out.view(*size).contiguous()


class XACANNet2s(nn.Module):
    def __init__(self, load_weights=False, uncertainty=False, in_channels=1024):
        super().__init__()
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=in_channels, batch_norm=True, dilation=True)
        # self.backend_short = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.Linear(1024, 20 if uncertainty else 10)
        # )
        # self.norm = nn.LayerNorm(64)
        self.output_layer = nn.Conv2d(64, 20 if uncertainty else 10, kernel_size=1)
        self.relu = nn.ReLU()
        self.xatt = AugmentedConv(512, 512, kernel_size=3, dk=512, dv=256, Nh=1)
        self.aggreg = None
        # self.xatt2 = AugmentedConv(512, 512, kernel_size=3, dk=512, dv=256)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self, x_prev, x, return_att=False):
        x_prev = self.frontend(x_prev)
        x = self.frontend(x)

        x_prev = self.context(x_prev)
        x = self.context(x)

        xatt1, weights1 = self.xatt(x, x_prev)
        xatt2, weights2 = self.xatt(x_prev, x)
        # xatt1 + xatt2
        if self.aggreg is None:
            x = torch.cat((xatt1, xatt2), 1)
        else:
            x = self.aggreg(xatt1, xatt2)

        x = self.backend(x)

        x = self.output_layer(x)
        x = self.relu(x)

        if return_att:
            return x, (weights1, weights2)
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# initialize module's weights to zero
def zero(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.zeros_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)
