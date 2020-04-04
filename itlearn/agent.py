import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

from enc_rnn import RNNEnc
from dec_rnn import RNNDec
from dec_attn import RNNDecAttn

from utils.misc import cuda, normf
from modules import ArgsModule

SMALL = 1e-10

class RNN(ArgsModule):
    def __init__(self, args, voc_sz_src, voc_sz_trg):
        super(RNN, self).__init__(args)

        self.enc = RNNEnc(args, voc_sz_src)
        self.dec = RNNDec(args, voc_sz_trg)

    def forward(self, x, x_len, y):
        hid = self.enc(x, x_len) # (batch_size, num_dir_enc * D_hid_enc)
        logits = self.dec(hid, y)
        return logits, hid

    def decode(self, x, x_len, decode_method, beam_width):
        hid = self.enc(x, x_len) # (batch_size, num_dir_enc * D_hid_enc)
        if decode_method == "greedy":
            hyp = self.dec.send(hid, x_len, x_len, "argmax")['msg']
            hyp = hyp.cpu().numpy().tolist()
        elif decode_method == "beam":
            hyp = self.dec.beam_search(hid, x_len, beam_width)
        return hyp


class RNNAttn(ArgsModule):
    def __init__(self, args, voc_sz_src, voc_sz_trg):
        super(RNNAttn, self).__init__(args)

        self.enc = RNNEnc(args, voc_sz_src)
        self.dec = RNNDecAttn(args, voc_sz_trg)

    def forward(self, x, x_len, y):
        hid = self.enc(x, x_len) # (batch_size, x_seq_len, num_dir_enc * D_hid_enc)
        logits = self.dec(hid, x_len, y)
        return logits, hid

    def decode(self, x, x_len, decode_method, beam_width):
        hid = self.enc(x, x_len) # (batch_size, x_seq_len, num_dir_enc * D_hid_enc)
        if decode_method == "greedy":
            hyp = self.dec.send(hid, x_len, None, "argmax", None)['msg']
            hyp = hyp.cpu().numpy().tolist()
        elif decode_method == "beam":
            hyp = self.dec.beam(hid, x_len)
        return hyp


class ImageGrounding(ArgsModule):
    def __init__(self, args, voc_sz):
        super(ImageGrounding, self).__init__(args)
        self.emb = nn.Embedding(voc_sz, args.D_emb)
        #self.rnn = nn.LSTM(args.D_emb, args.D_hid, args.n_layers, dropout=args.drop_ratio,
        #                   batch_first=True, bidirectional=True)
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.n_layers, dropout=args.drop_ratio,
                          batch_first=True)
        if args.img_pred_loss == "vse":
            self.img_enc = Linear(args.D_img, args.D_hid)
        elif args.img_pred_loss == "mse":
            self.img_enc = Linear(args.D_hid, args.D_img)
        else:
            raise Exception

        self.voc_sz = voc_sz
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.img_enc.weight.data, gain=nn.init.calculate_gain('linear'))
        nn.init.constant_(self.img_enc.bias.data, 0)
        nn.init.uniform_(self.emb.weight.data, -0.1, 0.1)
        print("Initialised!")

    def forward(self, x, x_len, img):
        # x : (batch_size, seq_len)
        # x_len : (batch_size)
        # img : (batch_size, D_img)
        batch_size, seq_len = x_len.size(0), x_len.max().item()
        x_enc = self.get_cap_rep(x, x_len)
        R = {}

        if self.img_pred_loss == "mse":
            x_enc = F.dropout( x_enc, p=self.drop_ratio, training=self.training ) # (batch_size, D_img)
            x_enc = self.img_enc( x_enc )
            #loss = F.mse_loss(x_enc, img, reduction='none')
            loss = ( (x_enc - img) ** 2 ).mean(1)

        elif self.img_pred_loss == "vse":
            img = F.dropout( img, p=self.drop_ratio, training=self.training ) # (batch_size, D_img)
            img = self.img_enc( img ) # (batch_size, D_hid)

            x_enc = normf(x_enc)
            img = normf(img)

            scores = torch.mm( img, x_enc.t() ) # (batch_size, batch_size)
            diagonal = scores.diag().view(batch_size, -1) # (batch_size, 1)
            pos_cap_scores = diagonal.expand_as(scores) # (batch_size, batch_size)
            pos_img_scores = diagonal.t().expand_as(scores) # (batch_size, batch_size)

            cost_cap = (self.margin + scores - pos_cap_scores).clamp(min=0)
            cost_img = (self.margin + scores - pos_img_scores).clamp(min=0)

            mask = cuda( torch.eye( batch_size ) > .5 ) # remove diagonal
            cost_cap = cost_cap.masked_fill(mask, 0.0) # (batch_size, batch_size)
            cost_img = cost_img.masked_fill(mask, 0.0) # (batch_size, batch_size)

            if self.training:
                loss = cost_cap.max(1)[0] + cost_img.max(0)[0] # (batch_size)
            else:
                loss = cost_cap.mean(dim=1) + cost_img.mean(dim=0) # (batch_size)
        R['loss'] = loss
        return R

    def get_cap_rep(self, x, x_len):
        batch_size, seq_len = x.shape
        hidden = cuda(torch.zeros(self.n_layers, batch_size, self.D_hid))
        emb = F.dropout( self.emb( x ), p=self.drop_ratio, training=self.training )

        output, _ = self.rnn(emb, hidden)
        # (batch, seq_len, 2 * D_hid)
        f_out = output.view(batch_size, seq_len, self.D_hid)
        f_idx = (x_len - 1)[:,None,None].repeat(1, 1, self.D_hid)
        f_out = f_out.gather(dim=1, index=f_idx).view(batch_size, -1) # (batch_size, D_hid)
        return f_out

    def batch_enc_img(self, img_feat):
        """
        :param img_feat: [NB_img, D_img]
        :return: [NB_img, D_hid]
        """
        batch_size = 64
        result = []
        start = 0
        while start < img_feat.shape[0]:
            end = start + batch_size
            batch_img_feat = cuda(img_feat[start: end])
            batch_img_feat = F.dropout(batch_img_feat,
                                       p=self.drop_ratio,
                                       training=self.training )
            batch_img_feat = self.img_enc(batch_img_feat)
            result.append(batch_img_feat)
            start = end
        result = torch.cat(result, dim=0)
        return normf(result)

    def batch_cap_rep(self, sents, sent_lens):
        """
        :param sents: [NB_SENT, len]
        :param sent_lens: [NB_SENT]
        :return: [NB_X, D_hid]
        """
        batch_size = 64
        start = 0
        result = []
        while start < sents.shape[0]:
            end = start + batch_size
            batch_sent = cuda(sents[start: end])
            batch_len = cuda(sent_lens[start: end])
            sent_enc = self.get_cap_rep(batch_sent, batch_len)
            result.append(sent_enc)
            start = end
        result = torch.cat(result, dim=0)
        return normf(result)

class RNNLM(ArgsModule):
    def __init__(self, args, voc_sz):
        super(RNNLM, self).__init__(args)
        self.drop = nn.Dropout(args.drop_ratio)
        self.encoder = nn.Embedding(voc_sz, args.D_emb)
        self.rnn = nn.LSTM(args.D_emb, args.D_hid, args.n_layers, dropout=args.drop_ratio, batch_first=False)
        self.decoder = nn.Linear(args.D_hid, voc_sz)

        if args.tie_emb:
            if args.D_hid != args.D_emb:
                raise ValueError
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.voc_sz = voc_sz

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, target):
        seq_len, batch_size = input.shape
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output).view( seq_len * batch_size, self.voc_sz)
        return decoded, hidden

    def get_nll(self, en_msg):
        with torch.no_grad():
            batch_size = en_msg.size(0)
            data = en_msg.t().contiguous() # (seq_len+1, batch_size)
            input, target = data[:-1,:], data[1:,:] # (seq_len, batch_size), (seq_len, batch_size)
            hidden = self.init_hidden(batch_size)
            decoded, _ = self.forward(input, hidden, target)
            loss = F.cross_entropy( decoded, target.view(-1), ignore_index=0, reduction='none' )
            loss = loss.view(-1, batch_size)
            loss = loss.t().contiguous() # (batch_size, seq_len-1)
            return loss

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, bsz, self.D_hid),
                weight.new_zeros(self.n_layers, bsz, self.D_hid))

class ImageCaptioning(ArgsModule):
    def __init__(self, args, voc_sz):
        super(ImageCaptioning, self).__init__(args)
        if not self.no_img:
            self.img_enc = Linear(args.D_img, args.D_hid)
        self.encoder = nn.Embedding(voc_sz, args.D_emb)
        self.rnn = nn.LSTM(args.D_emb, args.D_hid, args.n_layers, dropout=args.drop_ratio, batch_first=True)
        self.decoder = nn.Linear(args.D_hid, voc_sz)

        self.voc_sz = voc_sz

    def forward(self, msg, img):
        # msg : (batch_size, seq_len) or (batch_size, seq_len, vocab_size)
        # img : (batch_size, D_img)
        batch_size = msg.size(0)
        if self.no_img:
            assert (img is None)
            hidden = cuda( torch.zeros(self.n_layers, batch_size, self.D_hid) )
        else:
            assert not (img is None)
            img = F.dropout( img, p=self.drop_ratio, training=self.training )
            hidden = self.img_enc( img )[None,:,:] # (1, batch_size, D_hid)
            hidden = hidden.repeat(self.n_layers, 1, 1)
        input, target = msg[:,:-1], msg[:,1:]
        if len(input.shape) == 2:
            emb = F.dropout(self.encoder( input ), p=self.drop_ratio, training=self.training)
        elif len(input.shape) == 3:
            emb = torch.matmul(input, self.encoder.weight)
            emb = F.dropout(emb, p=self.drop_ratio, training=self.training)
        else:
            raise ValueError
        output, _ = self.rnn(emb, (hidden, hidden))
        output = F.dropout( output, p=self.drop_ratio, training=self.training )
        decoded = self.decoder(output).view(-1, self.voc_sz)
        return decoded

    def get_loss(self, msg, img):
        with torch.no_grad():
            batch_size = msg.size(0)
            input, target = msg[:,:-1], msg[:,1:].contiguous() # (batch_size, seq_len) x 2
            decoded = self.forward( msg, img )
            loss = F.cross_entropy( decoded, target.view(-1), ignore_index=0, reduction='none' )
            loss = loss.view(batch_size, -1) # (batch_size, seq_len)
        return loss

    def get_loss_oh(self, msg_oh, img):
        """
        :param msg_oh: [bsz, len, vocab_size]
        """
        batch_size = msg_oh.size(0)
        input, target = msg_oh[:, :-1], msg_oh[:, 1:].contiguous()  # (batch_size, seq_len) x 2
        target = torch.argmax(target, dim=-1)
        decoded = self.forward(msg_oh, img)
        loss = F.cross_entropy(decoded, target.view(-1), ignore_index=0, reduction='none')
        loss = loss.view(batch_size, -1)  # (batch_size, seq_len)
        return loss

