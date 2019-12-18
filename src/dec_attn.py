import ipdb
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
        # key: (batch_size, D_key)
        # query: (batch_size, x_seq_len, D_query)
        # encoder_padding_mask : (batch_size, x_seq_len)

        # x: (batch_size, D_query)
        x = self.W_k(key)

        # compute attention
        attn_scores = (query * x[:,None,:]).sum(dim=2) # (batch_size, x_seq_len)

        attn_scores = attn_scores.float().masked_fill_(
            encoder_padding_mask.bool(),
            float('-inf')
        )

        attn_scores = F.softmax(attn_scores, dim=1) # (batch_size, x_seq_len)

        x = (attn_scores[:,:,None] * query).sum(dim=1) # (batch_size, D_query)

        x = torch.tanh( self.W_q( torch.cat( [x, key], dim=1) ) )
        return x, attn_scores # batch_size x D_query


class RNNDecAttn(ArgsModule):
    def __init__(self, args, voc_sz_trg):
        super(RNNDecAttn, self).__init__(args)

        self.voc_sz_trg = voc_sz_trg
        self.emb = nn.Embedding(voc_sz_trg, args.D_emb, padding_idx=0)

        self.layers = nn.ModuleList([\
            nn.GRUCell(
                input_size = args.D_emb + args.D_hid if layer == 0 and args.input_feeding else args.D_emb,
                hidden_size = args.D_hid,
            )
            for layer in range(args.n_layers)
        ])

        self.attention = AttentionLayer(args.D_hid, args.D_hid) if self.model == "RNNAttn" else None

        self.out = nn.Linear(args.D_hid, voc_sz_trg)
        #if args.tie_emb and (args.D_hid == args.D_emb):
        #    self.out.weight = self.emb.weight

    def forward(self, src_hid, src_len, trg_tok):
        # src_hid : (batch_size, x_seq_len, D_hid * n_dir)
        # src_len : (batch_size)
        # trg_tok : (batch_size, y_seq_len)
        batch_size, x_seq_len = src_hid.size()[:2]
        src_mask = xlen_to_inv_mask(src_len, seq_len=x_seq_len) # (batch_size, x_seq_len)
        y_seq_len = trg_tok.size()[1]

        h_index = (src_len - 1)[:,None,None].repeat(1, 1, src_hid.size(2))
        z_t = torch.cumsum(src_hid, dim=1) # (batch_size, x_seq_len, D_hid * n_dir)
        z_t = z_t.gather(dim=1, index=h_index).view(batch_size, -1) # (batch_size, D_hid * n_dir)
        z_t = torch.div( z_t, src_len[:, None].float() ) # (batch_size, D_hid)

        y_emb = self.emb(trg_tok) # (batch_size, y_seq_len, D_emb)
        y_emb = F.dropout( y_emb, p=self.drop_ratio, training=self.training )

        outs = []
        prev_trg_hids = [z_t for i in range(self.n_layers)]
        input_feed = y_emb.data.new(batch_size, self.D_hid).zero_()

        for idx in range(y_seq_len):
            if self.input_feeding:
                input = torch.cat([ y_emb[:,idx,:], input_feed ], dim=1) # (batch_size, D_emb + D_hid)
            else:
                input = y_emb[:,idx,:]

            for i, rnn in enumerate(self.layers):
                trg_hid = rnn(input, prev_trg_hids[i]) # (batch_sizem D_hid)
                input = F.dropout(trg_hid, p=self.drop_ratio, training=self.training)
                prev_trg_hids[i] = trg_hid

            if self.attention is not None:
                out, attn_scores = self.attention(trg_hid, src_hid, src_mask) # (batch_size, D_hid)
            else:
                out = trg_hid

            input_feed = out
            outs.append(out)

        x = torch.cat(outs, dim=1).view(batch_size, y_seq_len, self.D_hid)
        x = self.out(x).view(-1, self.voc_sz_trg)

        return x

    def send(self, src_hid, src_len, trg_len, send_method, value_fn=None):
        # src_hid : (batch_size, x_seq_len, D_hid * n_dir)
        # src_len : (batch_size)
        batch_size, x_seq_len = src_hid.size()[:2]
        src_mask = xlen_to_inv_mask(src_len, seq_len=x_seq_len) # (batch_size, x_seq_len)
        y_seq_len = math.floor(trg_len.max().item() * self.msg_len_ratio) \
            if self.msg_len_ratio > 0 else self.max_len_gen

        h_index = (src_len - 1)[:,None,None].repeat(1, 1, src_hid.size(2))
        z_t = torch.cumsum(src_hid, dim=1) # (batch_size, x_seq_len, D_hid * n_dir)
        z_t = z_t.gather(dim=1, index=h_index).view(batch_size, -1) # (batch_size, D_hid * n_dir)
        z_t = torch.div(z_t, src_len[:, None].float())

        y_emb = cuda( torch.full((batch_size,), self.init_token).long() )
        y_emb = self.emb( y_emb ) # (batch_size, D_emb)
        #y_emb = F.dropout( y_emb, p=self.drop_ratio, training=self.training )

        prev_trg_hids = [z_t for _ in range(self.n_layers)]
        trg_hid = prev_trg_hids[0]
        input_feed = y_emb.data.new(batch_size, self.D_hid).zero_()

        done = cuda( torch.zeros(batch_size).long() )
        seq_lens = cuda( torch.zeros(batch_size).fill_(y_seq_len).long() ) # (max_len)
        max_seq_lens = cuda( torch.zeros(batch_size).fill_(y_seq_len).long() ) # (max_len)
        eos_tensor = cuda( torch.zeros(batch_size).fill_( self.eos_token )).long()

        self.log_probs, self.R_b, self.neg_Hs = [], [], []
        msg = []

        for idx in range(y_seq_len):
            if self.input_feeding:
                input = torch.cat([ y_emb, input_feed ], dim=1) # (batch_size, D_emb + D_hid)
            else:
                input = y_emb

            if send_method == "reinforce" and value_fn:
                if self.input_feeding:
                    value_fn_input = [ y_emb, input_feed ]
                else:
                    value_fn_input = [ y_emb, trg_hid ]
                value_fn_input = torch.cat( value_fn_input, dim=1 ).detach()
                self.R_b.append( value_fn( value_fn_input ).view(-1) ) # (batch_size)

            for i, rnn in enumerate(self.layers):
                trg_hid = rnn(input, prev_trg_hids[i]) # (batch_size, D_hid)
                #input = F.dropout(trg_hid, p=self.drop_ratio, training=self.training)
                input = trg_hid
                prev_trg_hids[i] = trg_hid

            if self.attention is not None:
                out, attn_scores = self.attention(trg_hid, src_hid, src_mask) # (batch_size, D_hid)
            else:
                out = trg_hid

            input_feed = out
            logit = self.out(out) # (batch_size, voc_sz_trg)

            #if send_method == "reinforce" and idx < self.min_len_gen:
            #    logit[:, self.eos_token] = float('-inf') # NOTE force not to generate messages too short

            if send_method == "argmax":
                tokens = logit.max(dim=1)[1] # (batch_size)

            elif send_method == "reinforce":
                tok_dist = Categorical(logits=logit)
                tokens = tok_dist.sample()
                self.log_probs.append( tok_dist.log_prob(tokens) ) # (batch_size)

                #if idx >= self.min_len_gen:
                self.neg_Hs.append( -1 * tok_dist.entropy() )
            else:
                raise ValueError

            msg.append(tokens)
            is_next_eos = (tokens == eos_tensor ).long() # (batch_size)
            new_seq_lens = max_seq_lens.clone().masked_fill_(mask=is_next_eos.bool(),
                                                             value=float(idx+1)) # NOTE idx+1 ensures this is valid length
            seq_lens = torch.min(seq_lens, new_seq_lens) # (contains lengths)

            done = (done + is_next_eos).clamp(min=0, max=1).long()
            if done.sum() == batch_size:
                break

            y_emb = self.emb(tokens) # (batch_size, D_emb)

        msg = torch.stack(msg, dim=1) # (batch_size, y_seq_len)
        if send_method == "reinforce":
            self.log_probs = torch.stack(self.log_probs, dim=1) # (batch_size, y_seq_len)
            self.R_b = torch.stack(self.R_b, dim=1) if value_fn else self.R_b # (batch_size, y_seq_len)
            self.neg_Hs = torch.stack(self.neg_Hs, dim=1) # (batch_size, y_seq_len)

        result = {"msg":msg.clone(), "new_seq_lens":seq_lens.clone()}
        # NOTE en_msg_len = min( en_ref_len, whenever the model decides to output <EOS> symbol )
        if self.msg_len_ratio > 0:
            en_ref_len = torch.floor( trg_len.float() * self.msg_len_ratio ).long()
            seq_lens = torch.max( torch.min( seq_lens, en_ref_len ) , \
                                 seq_lens.new(seq_lens.size()).fill_(self.min_len_gen) )
            #result['seq_lens'] = seq_lens

            # NOTE make sure the message ends in an <EOS> symbol
            # this is to make sure that rubbish messages get high EN LM NLL
            ends_with_eos = (msg.gather(dim=1, index=(seq_lens-1)[:,None]).view(-1) == eos_tensor ).long()
            # (1 if eos, 0 otherwise)
            eos_or_pad = ends_with_eos * self.pad_token + (1-ends_with_eos) * self.eos_token
            # (eos_token if eos, pad_token if otherwise)
            msg = torch.cat([msg, msg.new(batch_size, 1).fill_(0)], dim=1)
            # (batch_size, y_seq_len + 1)
            msg.scatter_(dim=1, index=seq_lens[:,None], src=eos_or_pad[:,None])
            seq_lens = seq_lens + (1-ends_with_eos)
            msg_mask = xlen_to_inv_mask(seq_lens, seq_len=msg.size(1)) # (batch_size, x_seq_len)
            msg.masked_fill_(msg_mask, self.eos_token)
            result.update({"msg":msg, "new_seq_lens":seq_lens})

        return result
        # msg : EN message
        # seq_lens : action seq_len (uttered by the decoder, may not include <EOS>)
        # new_seq_lens : seq_len seen by the next model (including <EOS>)

    def beam(self, src_hid, src_len):
        # src_hid : (batch_size, x_seq_len, D_hid * n_dir)
        # src_len : (batch_size)
        batch_size, x_seq_len = src_hid.size()[:2]
        src_mask = xlen_to_inv_mask(src_len, seq_len=x_seq_len) # (batch_size, x_seq_len)
        voc_size, width = self.voc_sz_trg, self.beam_width
        y_seq_len = self.max_len_gen

        h_index = (src_len - 1)[:,None,None].repeat(1, 1, src_hid.size(2))
        z_t = torch.cumsum(src_hid, dim=1) # (batch_size, x_seq_len, D_hid * n_dir)
        z_t = z_t.gather(dim=1, index=h_index).view(batch_size, -1) # (batch_size, D_hid * n_dir)
        z_t = torch.div( z_t, src_len[:, None].float() )

        y_emb = cuda( torch.full( (batch_size,), self.init_token).long() )
        y_emb = self.emb( y_emb ) # (batch_size, D_emb)

        input_feed = y_emb.data.new(batch_size, self.D_hid).zero_()

        live = [ [ ( 0.0, [ self.init_token ], 2 ) ] for ii in range(batch_size) ]
        dead = [ [] for ii in range(batch_size) ]
        n_dead = [0 for ii in range(batch_size)]
        src_hid_ = src_hid
        src_mask_ = src_mask

        for idx in range(y_seq_len):
            cwidth = 1 if idx == 0 else width
            input = torch.cat([ y_emb, input_feed ], dim=1) # (batch_size * width, D_emb + D_hid)
            trg_hid = self.layers[0](input, z_t) # (batch_size * width, D_hid)
            z_t = trg_hid # (batch_size * width, D_hid)

            out, attn_scores = self.attention(trg_hid, src_hid_, src_mask_) # (batch_size * width, D_hid)
            input_feed = out # (batch_size * width, D_hid)

            logit = self.out(out) # (batch_size * width, voc_sz_trg)
            cur_prob = F.log_softmax( logit, dim=1 ).view(batch_size, cwidth, self.voc_sz_trg)
            pre_prob = cuda( torch.FloatTensor( [ [ x[0] for x in ee ] for ee in live ] )\
                            .view(batch_size, cwidth, 1) ) # (batch_size, width, 1)
            total_prob = cur_prob + pre_prob # (batch_size, cwidth, voc_sz)
            total_prob = total_prob.view(batch_size, -1) # (batch_size, cwidth * voc_sz)

            topi_s = total_prob.topk(width, dim=1)[1]
            topv_s = cur_prob.view(batch_size, -1).gather(1, topi_s)

            new_live = [ [] for ii in range(batch_size) ]
            for bidx in range(batch_size):
                n_live = width - n_dead[bidx]
                if n_live > 0:
                    tis = topi_s[bidx][:n_live].cpu().numpy().tolist()
                    tvs = topv_s[bidx][:n_live].cpu().numpy().tolist()
                    for eidx, (topi, topv) in enumerate(zip(tis, tvs)):
                        if topi % voc_size == self.eos_token :
                            dead[bidx].append( ( live[bidx][ topi // voc_size ][0] + topv,
                                                 live[bidx][ topi // voc_size ][1] + [ topi % voc_size ],
                                                 topi ) )
                            n_dead[bidx] += 1
                        else:
                            new_live[bidx].append( ( live[bidx][ topi // voc_size ][0] + topv,
                                                     live[bidx][ topi // voc_size ][1] + [ topi % voc_size ],
                                                     topi ) )
                while len(new_live[bidx]) < width:
                    new_live[bidx].append( (    -99999999999,
                                                [0],
                                                0) )
            live = new_live
            if n_dead == [width for ii in range(batch_size)]:
                break

            in_vocab_idx = np.array( [ [ x[2] % voc_size for x in ee ] for ee in live ] ) # NOTE batch_size first
            y_emb = self.emb( cuda( torch.LongTensor( in_vocab_idx ) ).view(-1) )\
                    .view(-1, self.D_emb) # input (batch_size * width, 1, D_emb)

            in_width_idx = np.array( [ [ x[2] // voc_size + bbidx * cwidth for x in ee ] \
                            for bbidx, ee in enumerate(live) ] )
            in_width_idx = cuda( torch.LongTensor( in_width_idx ).view(-1) )
            z_t = z_t.index_select( 0, in_width_idx ).view(batch_size * width, self.D_hid)
            input_feed = input_feed.index_select( 0, in_width_idx ).view(batch_size * width, self.D_hid)
            src_hid_ = src_hid_.index_select( 0, in_width_idx ).view(batch_size * width, x_seq_len, self.D_hid)
            src_mask_ = src_mask_.index_select( 0, in_width_idx ).view(batch_size * width, x_seq_len)

        for bidx in range(batch_size):
            if n_dead[bidx] < width:
                for didx in range( width - n_dead[bidx] ):
                    (a, b, c) = live[bidx][didx]
                    dead[bidx].append( (a, b, c)  )

        dead_ = [ [ ( a / ( math.pow(5+len(b), self.beam_alpha) / math.pow(5+1, self.beam_alpha) ) , b, c)\
                   for (a,b,c) in ee] for ee in dead]
        ans = [ [], [], [], [], [] ]
        for dd_ in dead_:
            dd = sorted( dd_, key=operator.itemgetter(0), reverse=True )
            for idx in range(5):
                ans[idx].append( dd[idx][1] )
            #ans.append( dd[0][1] )
        return ans

    def multi_decode(self, src_hids, src_lens):
        # src_hid : (batch_size, x_seq_len, D_hid * n_dir)
        # src_len : (batch_size)
        batch_size = src_hids[0].size(0)
        x_seq_lens = [src_hid.size(1) for src_hid in src_hids]
        src_masks = [ xlen_to_inv_mask(src_len, seq_len=x_seq_len) for (src_len, x_seq_len) \
                     in zip(src_lens, x_seq_lens) ] # (batch_size, x_seq_len)
        y_seq_len = self.max_len_gen
        z_ts = []

        for src_len, src_hid in zip(src_lens, src_hids):
            h_index = (src_len - 1)[:,None,None].repeat(1, 1, src_hid.size(2))
            z_t = torch.cumsum(src_hid, dim=1) # (batch_size, x_seq_len, D_hid * n_dir)
            z_t = z_t.gather(dim=1, index=h_index).view(batch_size, -1) # (batch_size, D_hid * n_dir)
            z_t = torch.div( z_t, src_len[:, None].float() )
            z_ts.append( z_t )

        y_emb = cuda( torch.full((batch_size,), self.init_token).long() )
        y_emb = self.emb( y_emb ) # (batch_size, D_emb)
        input_feeds = [ y_emb.data.new(batch_size, self.D_hid).zero_() for ii in range(5) ]

        done = cuda( torch.zeros(batch_size).long() )
        eos_tensor = cuda( torch.zeros(batch_size).fill_( self.eos_token )).long()

        msg = []
        for idx in range(y_seq_len):
            probs = []
            for which in range(5):
                input = torch.cat([ y_emb, input_feeds[which] ], dim=1) # (batch_size * width, D_emb + D_hid)
                trg_hid = self.layers[0](input, z_ts[which]) # (batch_size * width, D_hid)
                z_ts[which] = trg_hid # (batch_size * width, D_hid)

                out, _ = self.attention(trg_hid, src_hids[which], src_masks[which]) # (batch_size, D_hid)
                input_feeds[which] = out
                logit = self.out(out) # (batch_size, voc_sz_trg)
                probs.append( F.softmax( logit, -1 ) )

            prob = torch.stack(probs, dim=-1).mean(-1)
            tokens = prob.max(dim=1)[1] # (batch_size)
            msg.append(tokens)

            is_next_eos = ( tokens == eos_tensor ).long() # (batch_size)
            done = (done + is_next_eos).clamp(min=0, max=1).long()
            if done.sum() == batch_size:
                break

            y_emb = self.emb(tokens) # (batch_size, D_emb)

        msg = torch.stack(msg, dim=1) # (batch_size, y_seq_len)
        return msg
