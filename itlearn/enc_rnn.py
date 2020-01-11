import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import cuda, xlen_to_inv_mask
from modules import ArgsModule

def take_last(output, x_len, n_dir, D_hid):
    batch_size = x_len.size()[0]
    x_seq_len = x_len.max().item()

    if n_dir == 1:
        index = (x_len - 1)[:,None,None].repeat(1, 1, output.size(2))
        # index : (batch_size, 1 , n_dir * D_hid)
        out = output.gather(dim=1, index=index).view(batch_size, -1)

    elif n_dir == 2:
        index = (x_len - 1)[:,None,None].repeat(1, 1, output.size(2) / 2)
        # index : (batch_size, 1 , D_hid)
        output = output.view(batch_size, x_seq_len, 2, D_hid)
        fwd = output[:,:,0,:].gather(dim=1, index=index).view(batch_size, -1) # (batch_size, D_hid)
        bwd = output[:,0,1,:] # (batch_size, D_hid)
        out = torch.cat([fwd, bwd], dim=1)

    return out # (batch_size, n_dir * D_hid)

class RNNEnc(ArgsModule):
    def __init__(self, args, voc_sz_src):
        super(RNNEnc, self).__init__(args)

        bi = True if args.n_dir == 2 else False
        self.emb = nn.Embedding(voc_sz_src, args.D_emb, padding_idx=0)
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.n_layers, \
                          batch_first=True, bidirectional=bi, dropout=args.drop_ratio)

    def forward(self, x, x_len, h_0=None):
        # NOTE x_len.max() == x.size(1)
        # x : (batch_size, x_seq_len) or (batcg_size, x_seq_len, vocab_size)
        # x_len : (batch_size)
        batch_size, x_seq_len = x.size()[:2] # NOTE dim==3 for gumbel softmax

        if h_0 is None:
            h_0 = cuda(torch.FloatTensor(self.n_layers * self.n_dir,
                                         batch_size, self.D_hid).zero_())
        if len(x.shape) == 2:
            input = self.emb(x)
        elif len(x.shape) == 3:
            # Gumbel
            input = torch.matmul(x, self.emb.weight)
        else:
            raise ValueError
        input = F.dropout( input, p=self.drop_ratio, training=self.training )

        # input (batch_size, x_seq_len, D_emb)
        # h_0 (n_layers * n_dir, batch_size, D_hid)
        output, _ = self.rnn(input, h_0)
        # output (batch_size, x_seq_len, n_dir * D_hid)
        # h_n (n_layers * n_dir, batch_size, D_hid)

        """
        if self.model == "RNN" : # RNN
            out = take_last(output, x_len, self.n_dir, self.D_hid)
            return out # (batch_size, n_dir * D_hid)

        else: # RNNAttn
            #inv_mask = xlen_to_inv_mask(x_len)[:,:,None] # (batch_size, x_seq_len, 1)
            #output.masked_fill_(inv_mask, 0)
        """
        return output.contiguous() # (batch_size, x_seq_len, n_dir * D_hid)

