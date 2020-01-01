import ipdb
import operator
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import cuda, gumbel_softmax, gumbel_softmax_hard
from modules import ArgsModule

class RNNDec(ArgsModule):
    def __init__(self, args):
        super(RNNDec, self).__init__(args)

        self.emb = nn.Embedding(args.voc_sz_trg, args.D_emb, padding_idx=0)
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.n_layers, \
                          dropout=args.drop_ratio, batch_first=True)

        self.ctx_to_h0 = nn.Linear(args.n_dir * args.D_hid, args.n_layers * args.D_hid)
        self.out = nn.Linear(args.D_hid, args.voc_sz_trg, bias=True)

        if args.tie_emb:
            self.emb.weight = self.out.weight

    def forward(self, h_ctx, y):
        # h_ctx : (batch_size, n_dir * D_hid )
        # y : (batch_size, y_seq_len)
        batch_size, y_seq_len = y.size()

        h_0 = self.ctx_to_h0(h_ctx).view(batch_size, self.n_layers, self.D_hid)\
                .transpose(0,1).contiguous()
        input = self.emb( y )
        input = F.dropout( input, p=self.drop_ratio, training=self.training )

        # input (batch_size, y_seq_len, D_emb)
        # h_0 (n_layers, batch_size, D_hid)
        output, _ = self.rnn(input, h_0)
        # output (batch_size, y_seq_len, D_hid)
        # h_n (n_layers, batch_size, D_hid)

        logits = self.out( output.contiguous().view(-1, self.D_hid) ) # (batch_size * y_seq_len, voc_sz_trg)

        return logits

    def send(self, h_ctx, h_len, y_len, send_method):
        # h_ctx : (batch_size, n_dir * D_hid)
        batch_size = h_ctx.size()[0]

        y_seq_len = int(trg_len.max().item() * self.msg_len_ratio) if self.msg_len_ratio > 0 else self.max_len_gen

        done = cuda( torch.zeros(batch_size).long() )
        seq_lens = cuda( torch.zeros(batch_size).fill_(y_seq_len).long() ) # (max_len)
        max_seq_lens = cuda( torch.zeros(batch_size).fill_(y_seq_len).long() ) # (max_len)
        eos_tensor = cuda( torch.zeros(batch_size).fill_( self.eos_token )).long()

        input = cuda( torch.full((batch_size, 1), self.init_token ).long() ) # (batch_size, 1)
        input = self.emb( input )
        msg = []
        self.log_probs = [] # (batch_size, seq_len)

        hid = self.ctx_to_h0(h_ctx).view(batch_size, self.n_layers, self.D_hid)\
                .transpose(0,1).contiguous()

        batch_logits = 0

        for idx in range(y_seq_len):
            # input (batch_size, 1, D_emb)
            # hid (n_layers, batch_size, D_hid)
            output, hid = self.rnn( input, hid )
            # output (batch_size, 1, D_hid)
            # hid (n_layers, batch_size, D_hid)

            output = self.out( output.view(-1, self.D_hid) )
            # (batch_size, voc_sz_trg)

            if send_method == "argmax":
                tokens = output.data.max(dim=1)[1] # (batch_size)

            elif send_method == "reinforce":
                cat = Categorical(logits=output)
                tokens = cat.sample()
                self.log_probs.append( cat.log_prob(tokens) )

                if self.entropy:
                    batch_logits += (output * (1-done)[:,None].float()).sum(dim=0)

            msg.append(tokens.unsqueeze(dim=1))

            is_next_eos = ( tokens == eos_tensor ).long() # (batch_size)
              # (1 if eos, 0 otherwise)  
            done = (done + is_next_eos).clamp(min=0, max=1).long()
            new_seq_lens = max_seq_lens.clone().masked_fill_(mask=is_next_eos.byte(), \
                                                             value=float(idx+1)) # either max or idx if next is eos
            # max_seq_lens : (batch_size)
            seq_lens = torch.min(seq_lens, new_seq_lens)

            if done.sum() == batch_size:
                break

            input = self.emb(tokens)[:,None,:] # (batch_size, 1, D_emb)

        msg = torch.cat(msg, dim=1)
        if send_method == "reinforce": # want to sum per-token log prob to yield log prob for the whole message sentence
            self.log_probs = torch.stack(self.log_probs, dim=1) # (batch_size, seq_len)

          # (batch_size, seq_len) if argmax or reinforce
          # (batch_size, seq_len, voc_sz_trg) if gumbel
        results = {"msg":msg, "seq_lens":seq_lens}
        if send_method == "reinforce" and self.entropy:
            results.update( {"batch_logits":batch_logits} )
        return results

    def beam_search(self, h_ctx, h_len, width): # (batch_size, n_dir * D_hid)
        voc_size, batch_size = self.voc_sz_trg, h_ctx.size()[0]
        live = [ [ ( 0.0, [ self.init_token ], 0 ) ] for ii in range(batch_size) ]
        # live : a list of 3-tuples
        #    a : cumulative log prob
        #    b : list of tokens
        #    c : last beam index
        dead = [ [] for ii in range(batch_size) ]
        n_dead = [0 for ii in range(batch_size)]

        y_seq_len = int(trg_len.max().item() * self.msg_len_ratio) if self.msg_len_ratio > 0 else self.max_len_gen

        input = cuda( torch.full((batch_size, 1), self.init_token ).long() ) # (batch_size, 1)
        input = self.emb( input )

        hid = self.ctx_to_h0(h_ctx).view(batch_size, self.n_layers, self.D_hid).transpose(0,1)\
                .contiguous()

        for tidx in range(y_seq_len):
            # input (batch_size * width, 1, D_emb)
            # h_0 (n_layers, batch_size * width, D_hid)
            output, hid = self.rnn( input, hid )
            # output (batch_size * width, 1, D_hid)
            # h_n (n_layers, batch_size * width, D_hid)

            cur_prob = F.log_softmax( self.out( output.view(-1, self.D_hid) ), dim=1 )\
                    .view(batch_size, -1, voc_size) # (batch_size, width * voc_sz_trg)
            pre_prob = cuda( torch.FloatTensor( [ [ x[0] for x in ee ] for ee in live ] ).view(batch_size, -1, 1) ) \
                    # (batch_size, width * 1)
            total_prob = cur_prob + pre_prob # (batch_size, width * voc_size)
            total_prob = total_prob.view(batch_size, -1)

            _, topi_s = total_prob.topk( width, dim=1)
            topv_s = cur_prob.view(batch_size, -1).gather(1, topi_s)
            # (batch_size, width)

            new_live = [ [] for ii in range(batch_size) ]
            for bidx in range(batch_size):
                n_live = width - n_dead[bidx]
                if n_live > 0:
                    tis = topi_s[bidx][:n_live].cpu().numpy().tolist()
                    tvs = topv_s[bidx][:n_live].cpu().numpy().tolist()
                    for eidx, (topi, topv) in enumerate(zip(tis, tvs)): # NOTE max width times
                        if topi % voc_size == self.eos_token :
                            dead[bidx].append( (  live[bidx][ topi // voc_size ][0] + topv, \
                                                  live[bidx][ topi // voc_size ][1] + [ topi % voc_size ],\
                                                  topi) )
                            n_dead[bidx] += 1
                        else:
                            new_live[bidx].append( (    live[bidx][ topi // voc_size ][0] + topv, \
                                                        live[bidx][ topi // voc_size ][1] + [ topi % voc_size ],\
                                                        topi) )
                while len(new_live[bidx]) < width:
                    new_live[bidx].append( (    -99999999, \
                                                [0],\
                                                0) )
            live = new_live

            if n_dead == [width for ii in range(batch_size)]:
                break

            in_vocab_idx = cuda( torch.LongTensor( [ [ x[2] % voc_size for x in ee ] for ee in live ] ) )# NOTE batch_size first
            input = self.emb( in_vocab_idx.view(-1) ).view(-1, 1, self.D_emb)
                # input (batch_size * width, 1, D_emb)

            bb = 1 if tidx == 0 else width
            in_width_idx = cuda( torch.LongTensor( [ [ x[2] // voc_size + bbidx * bb for x in ee ] for bbidx, ee in enumerate(live) ] ) ) \
                    # live : (batch_size, width)
            hid = hid.index_select( 1, in_width_idx.view(-1) ).view(self.n_layers, -1, self.D_hid)
            # h_0 (n_layers, batch_size * width, D_hid)

        for bidx in range(batch_size):
            if n_dead[bidx] < width:
                for didx in range( width - n_dead[bidx] ):
                    (a, b, c) = live[bidx][didx]
                    dead[bidx].append( (a, b, c)  )

        dead_ = [ [ ( a / ( math.pow(5+len(b), self.beam_alpha) / math.pow(5+1, self.beam_alpha) ) , b, c) for (a,b,c) in ee] for ee in dead]
        ans = []
        for dd_ in dead_:
            dd = sorted( dd_, key=operator.itemgetter(0), reverse=True )
            ans.append( dd[0][1] )
        return ans

