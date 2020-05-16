import json


class Params():
    def __init__(self, json_path=None):
        if json_path:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update_from_json(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def update(self, parsed_args):
        assert isinstance(parsed_args, dict)
        self.__dict__.update(parsed_args)

    @classmethod
    def parse_cmd(cls, argv):
        res = {}
        ignore_args = []
        idx = 1
        while idx < len(argv):
            cur = argv[idx].strip()
            if cur.startswith("--") and idx + 1 < len(argv):
                nxt = argv[idx + 1].strip()
                if nxt.startswith("--") or cur in ignore_args:
                    print("IGNORED : {}".format(cur))
                    pass
                res[cur.replace("--", "")] = parse(nxt)
                idx += 1
            else:
                print("IGNORED : {}".format(cur))
            idx += 1
        return res, ignore_args

    def __str__(self):
        return ( ', '.join("{}: {}".format(k,v) for k, v in self.__dict__.items()) )

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def parse(val):
    if val == "None":
        return None
    elif val == "True" or val == "true":
        return True
    elif val == "False" or val == "false":
        return False
    elif isint(val):
        return int(val)
    elif isfloat(val):
        return float(val)
    else:
        return val

def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def itlearn_str(args):
    hp_str = "sameopt_{}_endefinetune{}_".format(args.same_opt, args.en_de_finetune) + \
             "frens2p{}_frenreset{}_".format(args.sil_s2p_ratio, args.fr_en_reset) + \
             "k1{}_".format(args.k1) + \
             "fren_k2{}_temp{}_lr{}_".format(args.fr_en_k2, args.fr_en_temp, args.fr_en_lr) + \
             "ende_k2{}_temp{}_lr{}_".format(args.en_de_k2, args.en_de_temp, args.en_de_lr) + \
             "itersteps_{}_".format(args.max_itlearn_steps)
    return hp_str


def get_hp_str(args):
    if args.setup == "joint":
        hp_str = "{}_".format(args.setup) + \
                 "seed{}_".format(args.seed) + \
                 "{}".format( "ce{}_pg{}_b{}_".format(args.ce_co, args.pg_co, args.b_co) ) +\
                 "{}".format( "h{}_".format(args.h_co) ) +\
                 "{}".format( "hann{}k_".format(args.h_co_anneal_steps // 1000) if args.h_co_anneal else "" ) +\
                 "{}".format( "lm{}_enlm{}_".format(args.en_lm_dataset, args.en_lm_nll_co) ) +\
                 "{}".format( "ranker{}_imgpred_{}_c{}_".format(args.ranker_dataset, args.img_pred_loss, args.img_pred_loss_co) ) +\
                 "lr{:.0e}_".format(args.lr) + \
                 "{}_".format(args.lr_anneal) + \
                 "ann{}k_".format(args.linear_anneal_steps // 1000) + \
                 "drop{}_".format(args.drop_ratio) + \
                 "ratio{}_".format(args.msg_len_ratio) + \
                 "s2pfreq{}_s2p{}_".format(args.s2p_freq, args.s2p_co) + \
                 "s2psteps{}_".format(args.s2p_steps if hasattr(args, 's2p_steps') else -1) + \
                 ""
    elif args.setup == 'itlearn':
        hp_str = "{}_".format(args.setup) + \
                 "seed{}_".format(args.seed) + \
                 "{}".format( "ce{}_pg{}_b{}_".format(args.ce_co, args.pg_co, args.b_co) ) +\
                 "{}".format( "h{}_".format(args.h_co) ) +\
                 "{}".format( "hann{}k_".format(args.h_co_anneal_steps // 1000) if args.h_co_anneal else "" ) +\
                 "{}".format( "lm{}_enlm{}_".format(args.en_lm_dataset, args.en_lm_nll_co) ) +\
                 "{}".format( "ranker{}_imgpred_{}_c{}_".format(args.ranker_dataset, args.img_pred_loss, args.img_pred_loss_co) ) +\
                 "lr{:.0e}_".format(args.lr) + \
                 "{}_".format(args.lr_anneal) + \
                 "ann{}k_".format(args.linear_anneal_steps // 1000) + \
                 "drop{}_".format(args.drop_ratio) + \
                 "ratio{}_".format(args.msg_len_ratio) + itlearn_str(args)
    elif args.setup == "gumbel":
        hp_str = "seed{}_".format(args.seed) + \
                 "lr{:.0e}_".format(args.lr) + \
                 "{}_".format(args.lr_anneal) + \
                 "ann{}k_".format(args.linear_anneal_steps // 1000) + \
                 "{}".format("h{}_".format(args.h_co)) + \
                 "{}".format("enlm{}_".format(args.en_lm_nll_co)) + \
                 "{}".format("imgpred{}_".format(args.img_pred_loss_co)) + \
                 "gtemp{}_".format(args.gumbel_temp) + \
                 "drop{}_".format(args.drop_ratio) + \
                 "ratio{}_".format( args.msg_len_ratio ) + \
                 "{}".format("clip{}_".format(args.grad_clip) if args.grad_clip != -1.0 else "") + \
                 "s2pfreq{}_s2p{}_".format(args.s2p_freq, args.s2p_co) + \
                 "s2psteps{}_".format(args.s2p_steps if hasattr(args, 's2p_steps') else -1) + \
                 ""
    elif args.setup == "gumbel_itlearn":
        hp_str = "seed{}_".format(args.seed) + \
                 "lr{:.0e}_".format(args.lr) + \
                 "gtemp{}_".format(args.gumbel_temp) + \
                 "s2pfreq{}_s2p{}_".format(args.s2p_freq, args.s2p_co) + \
                 "lenratio{}_".format(args.msg_len_ratio) + itlearn_str(args)

    elif args.setup == "single":
        hp_str = "{}_".format(args.setup) + \
                 "{}_".format(args.model.lower()) + \
                 "{}".format("{}_".format(args.pair)) + \
                 "emb{}_hid{}_".format(args.D_emb, args.D_hid) + \
                 "{}l_".format(args.n_layers) + \
                 "lr{:.0e}_".format(args.lr) + \
                 "{}_".format(args.lr_anneal) + \
                 "ann{}k_".format(args.linear_anneal_steps // 1000) + \
                 "drop{}_".format(args.drop_ratio) + \
                 "{}".format("clip{}_".format(args.grad_clip) if args.grad_clip != -1.0 else "") + \
                 ""
    elif args.setup == "ranker":
        hp_str = "{}_".format(args.setup) + \
                 "{}".format( "imgpred_{}_m{}_".format(args.img_pred_loss, args.margin) ) +\
                 "{}".format( "noimg{}_".format(args.no_img) if args.img_pred_loss == "nll" else "" ) +\
                 "img{}_emb{}_hid{}_".format(args.D_img, args.D_emb, args.D_hid) + \
                 "lr{:.0e}_".format(args.lr) + \
                 "{}_".format(args.lr_anneal) + \
                 "ann{}k_".format(args.linear_anneal_steps // 1000) + \
                 "drop{}_".format(args.drop_ratio) + \
                 "{}".format("clip{}_".format(args.grad_clip) if args.grad_clip != -1.0 else "") + \
                 ""

    elif args.setup == "lm":
        hp_str = "{}_".format(args.setup) + \
                 "{}_".format(args.dataset.lower()) + \
                 "emb{}_hid{}_".format(args.D_emb, args.D_hid) + \
                 "{}_".format(args.optimizer.lower()) + \
                 "drop{}_".format(args.drop_ratio) + \
                 "lr{:.0e}_".format(args.lr) + \
                 "anneal{}_".format(args.anneal_by) + \
                 "{}".format("clip{}_".format(args.grad_clip)) + \
                 ""
    return hp_str

