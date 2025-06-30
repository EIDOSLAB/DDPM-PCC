import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from models.gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, Grapher, DilatedKnnGraph, act_layer

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x += shortcut
        return x
    

class VIG(torch.nn.Module):
    def __init__(self, 
                 channels = 192,
                 k = 9,
                 act = 'relu',
                 norm = 'batch',
                 bias = True,
                 epsilon = 0.2,
                 stochastic = True,
                 conv = 'mr',
                 emb_dims = 256,
                 n_blocks = 12,
                 in_channels = 3,
                 block = 'grapher',
                 use_dilation = True):
        super(VIG, self).__init__()

        self.n_blocks = n_blocks

        
        drop_path = 0.0
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        # self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(in_channels, channels, conv, act, norm, bias=False)



        # if use_dilation:
        #     self.backbone = Seq(*[ResDynBlock2d(channels, k, i + 1, conv, act, norm,
        #                                         bias, stochastic, epsilon, knn)
        #                           for i in range(self.n_blocks - 1)])
        # else:
        #     self.backbone = Seq(*[ResDynBlock2d(channels, k, 1, conv, act, norm,
        #                                         bias, stochastic, epsilon, knn)
        #                           for _ in range(self.n_blocks - 1)])

        if use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = nn.Conv2d(channels, emb_dims, 1, bias=True)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1).unsqueeze(-1)
        x = self.head(inputs, self.knn(inputs))
        # x = self.stem(inputs) + self.pos_embed
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1), None




if __name__ == '__main__':
    device = torch.device('cuda')

    # feats = torch.rand((8, 3, 1024, 1), dtype=torch.float).to(device)
    feats = torch.rand((8,1024,3)).to(device)

    print('Input size {}'.format(feats.size()))
    net = VIG().to(device)
    out = net(feats)
    print(f'Output: {out.shape}')
    # print(net)
    # print('Output size {}'.format(out.size()))