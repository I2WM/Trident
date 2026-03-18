import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

# --------------------------------
# self-ensemble for test
# --------------------------------
def forward_x8(lr_img, forward_function=None):
    def _transform(v, op):
        # if args.precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(lr_img.device)
        # if args.precision == 'half': ret = ret.half()

        return ret

    self_ensemble_list = ['v', 'h', 't', 'hv', 'tv', 'th', 'thv']

    list_x = [lr_img]
    for tf in self_ensemble_list:
        tf_len=len(tf)
        if tf_len==1:
            list_x.extend([_transform(lr_img, tf)])
        elif tf_len==2:
            list_x.extend([_transform(_transform(lr_img, tf[0]),tf[1])])
        elif tf_len==3:
            list_x.extend([_transform(_transform(_transform(lr_img, tf[0]), tf[1]),tf[2])])

    list_y = []
    for x in list_x:
        y = forward_function(x)
        list_y.append(y)

    for i in range(len(list_y)):
        if i == 0:
            continue
        tf=self_ensemble_list[i - 1]
        tf_len=len(tf)
        if tf_len==1:
            list_y[i]=_transform(list_y[i], tf)
        elif tf_len==2:
            list_y[i]=_transform(_transform(list_y[i], tf[1]),tf[0])
        elif tf_len==3:
            list_y[i] =_transform(_transform(_transform(list_y[i], tf[2]), tf[1]),tf[0])

    y = [torch.cat(list_y, dim=0).mean(dim=0, keepdim=True)]

    if len(y) == 1: y = y[0]

    return y

@MODEL_REGISTRY.register()
class MambaIRModel(SRModel):
    """MambaIR model for image restoration."""

    # test by partitioning
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    # out = self.net_g_ema(chop)  # image processing of each partition
                    out = forward_x8(chop, self.net_g_ema.forward)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    # out = self.net_g(chop)  # image processing of each partition
                    out = forward_x8(chop, self.net_g.forward)   # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
