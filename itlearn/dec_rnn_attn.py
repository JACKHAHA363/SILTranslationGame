import operator
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, GRU, Embedding, Module
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import cuda, xlen_to_inv_mask, gumbel_softmax_hard
from modules import ArgsModule

class AttentionLayer(Module):
    def __init__(self, D_key, D_query):
        super(AttentionLayer, self).__init__()

        self.W_k = Linear(D_key, D_query, bias=False)
        self.W_q = Linear(D_key + D_query, D_query, bias=False)

    # General Attention in https://arxiv.org/pdf/1508.04025.pdf 
    def forward(self, key, query, encoder_padding_mask):
        # key: bsz x D_key
        # query: srclen x bsz x D_query

        # x: bsz x D_query
        x = self.W_k(key)

        # compute attention
        attn_scores = (query * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * query).sum(dim=0)

        x = F.tanh(self.W_q(torch.cat((x, key), dim=1)))
        return x, attn_scores

class RNNDecAttn(ArgsModule):
    def __init__(self, args):
        super(RNNDecAttn, self).__init__(args)

        self.emb            = nn.Embedding(args.voc_sz_trg, args.D_emb, padding_idx=0)

        self.w_hs           = nn.Linear(args.D_hid * args.n_dir, args.D_att)
        self.w_ht           = nn.Linear(args.D_hid, args.D_att)

        self.rnn            = nn.GRU(args.D_emb + args.D_hid, args.D_hid, \
                                args.n_layers, dropout=args.drop_ratio, batch_first=True)

        self.h_to_att       = nn.Linear(args.D_hid * args.n_dir, args.D_att)
        self.z_to_att       = nn.Linear(args.D_hid * args.n_layers, args.D_att)
        self.att_to_score   = nn.Linear(2 * args.D_hid, 1)

        self.c_to_out       = nn.Linear(args.D_hid, args.D_out)
        self.z_to_out       = nn.Linear(args.D_hid, args.D_out)
        self.out            = nn.Linear(args.D_out, args.voc_sz_trg)

        self.voc_size       = args.voc_sz_trg

        if args.D_out == args.D_emb and args.tie_emb:
            self.emb.weight = self.out.weight

    def forward(self, h, h_len, y):
        # h : (batch_size, x_seq_len, D_hid * n_dir)
        # h_len : (batch_size)
        # y : (batch_size, y_seq_len)
        batch_size, x_seq_len = h.size()[:2]
        y_seq_len = y.size()[1]
        xmask = xlen_to_inv_mask(h_len) # (batch_size, x_seq_len)

        # last hidden state of the reverse RNN
        h_index = (h_len - 1)[:,None,None].repeat(1, 1, h.size(2))
        z_t = torch.cumsum(h, dim=1) # (batch_size, x_seq_len, D_hid * n_dir)
        z_t = z_t.gather(dim=1, index=h_index).view(batch_size, -1) # (batch_size, D_hid * n_dir)
        z_t = torch.div( z_t, h_len[:, None].float() )
        z_t = self.hs_ht0( z_t ) # (batch_size, n_layers * D_hid)

        y_emb = self.emb(y) # (batch_size, y_seq_len, D_emb)
        y_emb = F.dropout( y_emb, p=self.drop_ratio, training=self.training )

        ctx_h = h # (batch_size, x_seq_len, D_hid * n_dir)

        logits = []
        for idx in range(y_seq_len):
            h_t_ = self.w_ht( self.w_ht ) # (batch_size, h_t)

            # in (batch_size, 1, D_emb)
            # z_t (n_layers, batch_size, D_hid)
            _, z_t_ = self.rnn( y_emb[:, idx:idx+1, :], z_t )
            # out (batch_size, 1, D_hid)
            # z_t (n_layers, batch_size, D_hid)
            ctx_z_t_ = z_t_.transpose(0,1).contiguous().view(batch_size, -1) \
                    # (batch_size, n_layers * D_hid)
            ctx_s = self.z_to_att( ctx_z_t_ )[:,None,:] # (batch_size, 1, D_att)
            ctx_y = self.y_to_att( y_emb[:, idx:idx+1, :] ) # (batch_size, 1, D_att)
            ctx = F.tanh(ctx_h + ctx_s + ctx_y) # (batch_size, x_seq_len, D_att)

            score = self.att_to_score(ctx).view(batch_size, -1) # (batch_size, x_seq_len)
            score.masked_fill_(xmask, -float('inf'))
            score = F.softmax( score, dim=1 )

            c_t = torch.mul( h, score[:,:,None] ) # (batch_size, x_seq_len, D_hid * n_dir)
            c_t = torch.sum( c_t, 1) # (batch_size, D_hid * n_dir)
            # in (batch_size, 1, D_hid * n_dir)
            # z_t (n_layers, batch_size, D_hid)
            out, z_t = self.rnn2( c_t[:,None,:], z_t_ )
            # out (batch_size, 1, D_hid)
            # z_t (n_layers, batch_size, D_hid)

            #fin_y = self.y_to_out( y_emb[:,idx,:] ) # (batch_size, D_out)
            fin_c = self.c_to_out( c_t ) # (batch_size, D_out)
            fin_s = self.z_to_out( out.view(-1, self.D_hid) ) # (batch_size, D_out)
            fin = F.tanh( fin_c + fin_s )

            logit = self.out( fin ) # (batch_size, voc_sz_trg)
            logits.append( logit )
            # logits : list of (batch_size, voc_sz_trg) vectors

        ans = torch.stack(logits, dim=1) # (batch_size, y_seq_len, voc_sz_trg)
        return ans.view(-1, ans.size(2))

    def send(self, h, h_len, y_len, send_method):
        # h : (batch_size, x_seq_len, D_hid * n_dir)
        # h_len : (batch_size)
        batch_size, x_seq_len = h.size()[:2]
        xmask = xlen_to_inv_mask(h_len) # (batch_size, x_seq_len)
        max_len_gen = self.max_len_gen if self.msg_len_ratio < 0.0 else int(y_len.max().item() * self.msg_len_ratio)
        max_len_gen = np.clip(max_len_gen, 2, self.max_len_gen).item()

        h_index = (h_len - 1)[:,None,None].repeat(1, 1, h.size(2))
        z_t = torch.cumsum(h, dim=1) # (batch_size, x_seq_len, D_hid * n_dir)
        z_t = z_t.gather(dim=1, index=h_index).view(batch_size, -1) # (batch_size, D_hid * n_dir)
        z_t = torch.div( z_t, h_len[:, None].float() )
        z_t = self.ctx_to_z0(z_t) # (batch_size, n_layers * D_hid)
        z_t = z_t.view(batch_size, self.n_layers, self.D_hid).transpose(0,1).contiguous() \

        y_emb = cuda( torch.full((batch_size, 1), self.init_token).long() )
        y_emb = self.emb( y_emb ) # (batch_size, 1, D_emb)
        y_emb = F.dropout( y_emb, p=self.drop_ratio, training=self.training )

        h_big = h.view(-1, h.size(2) ) \
                # (batch_size * x_seq_len, D_hid * n_dir) 
        ctx_h = self.h_to_att( h_big ).view(batch_size, x_seq_len, self.D_att)

        done = cuda( torch.zeros(batch_size).long() )
        seq_lens = cuda( torch.zeros(batch_size).fill_(max_len_gen).long() ) # (max_len)
        max_seq_lens = cuda( torch.zeros(batch_size).fill_(max_len_gen).long() ) # (max_len)
        eos_tensor = cuda( torch.zeros(batch_size).fill_( self.eoz_token )).long()
        msg = []
        self.log_probs = [] # (batch_size, seq_len)
        batch_logits = 0

        for idx in range(max_len_gen):
            # in (batch_size, 1, D_emb)
            # z_t (n_layers, batch_size, D_hid)
            _, z_t_ = self.rnn( y_emb, z_t )
            # out (batch_size, 1, D_hid)
            # z_t (n_layers, batch_size, D_hid)
            ctx_z_t_ = z_t_.transpose(0,1).contiguous().view(batch_size, self.n_layers * self.D_hid) \
                    # (batch_size, n_layers * D_hid)

            ctx_y = self.y_to_att( y_emb.view(batch_size, self.D_emb) )[:,None,:]
            ctx_s = self.z_to_att( ctx_z_t_ )[:,None,:]
            ctx = F.tanh(ctx_y + ctx_s + ctx_h)
            ctx = ctx.view(batch_size * x_seq_len, self.D_att)

            score = self.att_to_score(ctx).view(batch_size, -1) # (batch_size, x_seq_len)
            score.masked_fill_(xmask, -float('inf'))
            score = F.softmax( score, dim=1 )
            score = score[:,:,None] # (batch_size, x_seq_len, 1)

            c_t = torch.mul( h, score ) # (batch_size, x_seq_len, D_hid * n_dir)
            c_t = torch.sum( c_t, 1) # (batch_size, D_hid * n_dir)
            # in (batch_size, 1, D_hid * n_dir)
            # z_t (n_layers, batch_size, D_hid)
            out, z_t = self.rnn2( c_t[:,None,:], z_t_ )
            # out (batch_size, 1, D_hid)
            # z_t (n_layers, batch_size, D_hid)

            fin_y = self.y_to_out( y_emb.view(batch_size, self.D_emb) ) # (batch_size, D_out)
            fin_c = self.c_to_out( c_t ) # (batch_size, D_out)
            fin_s = self.z_to_out( out.view(batch_size, self.D_hid) ) # (batch_size, D_out)
            fin = F.tanh( fin_y + fin_c + fin_s )

            logit = self.out( fin ) # (batch_size, voc_sz_trg)

            if send_method == "argmax":
                tokens = logit.data.max(dim=1)[1] # (batch_size)
                tokens_idx = tokens

            elif send_method == "gumbel":
                tokens = gumbel_softmax_hard(logit, self.temp, self.st)\
                        .view(batch_size, self.voc_sz_trg) # (batch_size, voc_sz_trg)
                tokens_idx = (tokens * cuda(torch.arange(self.voc_sz_trg))[None,:]).sum(dim=1).long()

            elif send_method == "reinforce":
                cat = Categorical(logits=logit)
                tokens = cat.sample()
                tokens_idx = tokens
                self.log_probs.append( cat.log_prob(tokens) )

                if self.entropy:
                    batch_logits += (logit * (1-done)[:,None].float()).sum(dim=0)

            msg.append(tokens.unsqueeze(dim=1))

            is_next_eos = ( tokens_idx == eos_tensor ).long() # (batch_size)
              # (1 if eos, 0 otherwise)
            done = (done + is_next_eos).clamp(min=0, max=1).long()
            new_seq_lens = max_seq_lens.clone().masked_fill_(mask=is_next_eos.byte(), \
                                                             value=float(idx+1)) # either max or idx if next is eos
            # max_seq_lens : (batch_size)
            seq_lens = torch.min(seq_lens, new_seq_lens)

            if done.sum() == batch_size:
                break

            y_emb = self.emb(tokens)[:,None,:] # (batch_size, 1, D_emb)

        if self.msg_len_ratio > 0.0 :
            seq_lens = torch.clamp( (y_len.float() * self.msg_len_ratio).floor_(), 1, len(msg) ).long()

        msg = torch.cat(msg, dim=1)
        if send_method == "reinforce": # want to sum per-token log prob to yield log prob for the whole message sentence
            self.log_probs = torch.stack(self.log_probs, dim=1) # (batch_size, seq_len)

          # (batch_size, seq_len) if argmax or reinforce
          # (batch_size, seq_len, voc_sz_trg) if gumbel
        results = {"msg":msg, "seq_lens":seq_lens}
        if send_method == "reinforce" and self.entropy:
            results.update( {"batch_logits":batch_logits} )
        return results

    def beam_search(self, h, h_len, width): # (batch_size, x_seq_len, D_hid * n_dir)
        voc_size, batch_size, x_seq_len = self.voc_sz_trg, h.size()[0], h.size()[1]
        live = [ [ ( 0.0, [ self.init_token ], 2 ) ] for ii in range(batch_size) ]
        dead = [ [] for ii in range(batch_size) ]
        n_dead = [0 for ii in range(batch_size)]
        xmask = xlen_to_inv_mask(h_len)[:,None,:] # (batch_size, 1, x_seq_len)
        max_len_gen = self.max_len_gen if self.msg_len_ratio < 0.0 else int(x_seq_len * self.msg_len_ratio)
        max_len_gen = np.clip(max_len_gen, 2, self.max_len_gen).item()

        h_index = (h_len - 1)[:,None,None].repeat(1, 1, h.size(2))
        z_t = torch.cumsum(h, dim=1) # (batch_size, x_seq_len, D_hid * n_dir)
        z_t = z_t.gather(dim=1, index=h_index).view(batch_size, -1) # (batch_size, D_hid * n_dir)
        z_t = torch.div( z_t, h_len[:, None].float() )
        z_t = self.ctx_to_z0(z_t) # (batch_size, n_layers * D_hid)
        z_t = z_t.view(batch_size, self.n_layers, self.D_hid).transpose(0,1).contiguous() \

        input = cuda( torch.full((batch_size, 1), self.init_token).long() )
        input = self.emb( input ) # (batch_size, 1, D_emb)

        h_big = h.contiguous().view(-1, self.D_hid * self.n_dir ) \
                # (batch_size * x_seq_len, D_hid * n_dir)
        ctx_h = self.h_to_att( h_big ).view(batch_size, 1, x_seq_len, self.D_att) \
                # NOTE (batch_size, 1, x_seq_len, D_att)

        for tidx in range(max_len_gen):
            cwidth = 1 if tidx == 0 else width
            # input (batch_size * width, 1, D_emb)
            # z_t (n_layers, batch_size * width, D_hid)
            _, z_t_ = self.rnn( input, z_t )
            # out (batch_size * width, 1, D_hid)
            # z_t (n_layers, batch_size * width, D_hid)
            ctx_z_t_ = z_t_.transpose(0,1).contiguous().view(batch_size * cwidth, -1) \
                    # (batch_size * width, n_layers * D_hid)

            ctx_y = self.y_to_att( input.view(-1, self.D_emb) ).view(batch_size, cwidth, 1, self.D_att)
            ctx_s = self.z_to_att( ctx_z_t_ ).view(batch_size, cwidth, 1, self.D_att)
            ctx = F.tanh(ctx_y + ctx_s + ctx_h) # (batch_size, cwidth, x_seq_len, D_att)
            ctx = ctx.view(batch_size * cwidth * x_seq_len, self.D_att)

            score = self.att_to_score(ctx).view(batch_size, -1, x_seq_len) # (batch_size, cwidth, x_seq_len)
            score.masked_fill_(xmask.repeat(1, cwidth, 1), -float('inf'))
            score = F.softmax( score.view(-1, x_seq_len), dim=1 ).view(batch_size, -1, x_seq_len)
            score = score.view(batch_size, cwidth, x_seq_len, 1) # (batch_size, width, x_seq_len, 1)

            c_t = torch.mul( h.view(batch_size, 1, x_seq_len, -1), score ) # (batch_size, width, x_seq_len, D_hid * n_dir)
            c_t = torch.sum( c_t, 2).view(batch_size * cwidth, -1) # (batch_size * width, D_hid * n_dir)
            # c_t (batch_size * width, 1, D_hid * n_dir)
            # z_t (n_layers, batch_size * width, D_hid)
            out, z_t = self.rnn2( c_t[:,None,:], z_t_ )
            # out (batch_size * width, 1, D_hid)
            # z_t (n_layers, batch_size * width, D_hid)

            fin_y = self.y_to_out( input.view(-1, self.D_emb) ) # (batch_size * width, D_out)
            fin_c = self.c_to_out( c_t ) # (batch_size * width, D_out)
            fin_s = self.z_to_out( out.view(-1, self.D_hid) ) # (batch_size * width, D_out)
            fin = F.tanh( fin_y + fin_c + fin_s )

            cur_prob = F.log_softmax( self.out( fin.view(-1, self.D_out) ), dim=1 )\
                    .view(batch_size, cwidth, voc_size).data # (batch_size, width, voc_sz_trg)
            pre_prob = cuda( torch.FloatTensor( [ [ x[0] for x in ee ] for ee in live ] ).view(batch_size, cwidth, 1) ) \
                    # (batch_size, width, 1)
            total_prob = cur_prob + pre_prob # (batch_size, cwidth, voc_size)
            total_prob = total_prob.view(batch_size, -1)

            _, topi_s = total_prob.topk(width, dim=1)
            topv_s = cur_prob.view(batch_size, -1).gather(1, topi_s)
            # (batch_size, width)

            new_live = [ [] for ii in range(batch_size) ]
            for bidx in range(batch_size):
                n_live = width - n_dead[bidx]
                if n_live > 0:
                    tis = topi_s[bidx][:n_live]
                    tvs = topv_s[bidx][:n_live]
                    for eidx, (topi, topv) in enumerate(zip(tis, tvs)): # NOTE max width times
                        if topi % voc_size == self.eoz_token :
                            dead[bidx].append( (  live[bidx][ topi // voc_size ][0] + topv, \
                                                  live[bidx][ topi // voc_size ][1] + [ topi % voc_size ],\
                                                  topi) )
                            n_dead[bidx] += 1
                        else:
                            new_live[bidx].append( (    live[bidx][ topi // voc_size ][0] + topv, \
                                                        live[bidx][ topi // voc_size ][1] + [ topi % voc_size ],\
                                                        topi) )
                while len(new_live[bidx]) < width:
                    new_live[bidx].append( (    -99999999999, \
                                                [0],\
                                                0) )
            live = new_live

            if n_dead == [width for ii in range(batch_size)]:
                break

            in_vocab_idx = [ [ x[2] % voc_size for x in ee ] for ee in live ] # NOTE batch_size first
            input = self.emb( cuda( torch.LongTensor( in_vocab_idx ) ).view(-1) )\
                    .view(-1, 1, self.D_emb) # input (batch_size * width, 1, D_emb)

            in_width_idx = [ [ x[2] // voc_size + bbidx * cwidth for x in ee ] for bbidx, ee in enumerate(live) ] \
                    # live : (batch_size, width)
            z_t = z_t.index_select( 1, cuda( torch.LongTensor( in_width_idx ).view(-1) ) ).\
                    view(self.n_layers, batch_size * width, self.D_hid)
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

