from model import common
import torch.nn as nn

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv, transp_conv=common.transp_conv):
        super(EDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)
        
        # define upsample module
        m_upsample = [common.Slerp(scale)]

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        #####################
        # No tail required ??
        # ! SWITCHING OUT PIXEL SHUFFLE UPSAMPLER MODULE WITH A SLERP UPSAMPLER MODULE ! 
        #####################

        # m_tail = [
        #     common.Slerp
        # ]

        m_tail = [transp_conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.upsample = nn.Sequential(*m_upsample)

    # Forward method for the EDSR class
    def forward(self, x):
        import pdb; pdb.set_trace()
        x = self.head(x)
        res = self.body(x)
        res += x

        x = self.tail(res) # reduce number of channels from 128 to 4
        x_hr = self.upsample(x) # upsample with slerp
        import pdb; pdb.set_trace()
        return x_hr

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

