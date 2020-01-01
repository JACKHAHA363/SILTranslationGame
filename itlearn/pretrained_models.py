en_lm = {

    "wikitext103" : {
        "experiment": "180912_01_lm_wiki103",
        256 : "09.12_00.11._lm_emb256_hid256_sgd_reppen0.0_drop0.0_lr2e+01_anneal0.25_clip0.1_",
        512 : "09.12_00.11._lm_emb512_hid512_sgd_reppen0.0_drop0.0_lr2e+01_anneal0.25_clip0.1_",
    },

    "wikitext2": {
        "experiment": "180912_01_lm_wiki2",
        256 : "09.12_00.45._lm_emb256_hid256_sgd_reppen0.0_drop0.4_lr2e+01_anneal0.25_clip0.1_",
        512 : "09.12_00.45._lm_emb512_hid512_sgd_reppen0.0_drop0.4_lr2e+01_anneal0.25_clip0.1_",
    },

    "coco": {
        "experiment": "180914_02_ranker_nll_wo_img",
        256 : "09.14_14.48._ranker_imgpred_nll_m0_noimgTrue_img512_emb256_hid256_reppen0.0_lr1e-03_linear_ann100k_drop0.0_clip0.1_",
        512 : "09.14_14.48._ranker_imgpred_nll_m0_noimgTrue_img512_emb512_hid512_reppen0.0_lr3e-04_linear_ann100k_drop0.0_clip0.1_",
    },

    "multi30k": {
        "experiment": "180920_02_lm_flickr30k",
        256 : "09.20_18.59._ranker_multi30k_imgpred_nll_m0_noimgTrue_img2048_emb256_hid256_lr1e-03_linear_ann100k_drop0.3_clip0.1_",
        512 : "09.20_18.59._ranker_multi30k_imgpred_nll_m0_noimgTrue_img2048_emb512_hid512_lr1e-03_linear_ann100k_drop0.5_clip0.1_",
    },
}

ranker = {
    "coco":{
        "nll": {
            "experiment": "180914_02_ranker_nll_wo_img",
            256 : "09.14_14.48._ranker_imgpred_nll_m0_noimgFalse_img512_emb256_hid256_reppen0.0_lr1e-03_linear_ann100k_drop0.0_clip0.1_",
            512 : "09.14_14.48._ranker_imgpred_nll_m0_noimgFalse_img512_emb512_hid512_reppen0.0_lr3e-04_linear_ann100k_drop0.0_clip0.1_",
        },
        "vse":{
            "experiment": "180914_04_ranker_152",
            2048 : "09.14_21.08._ranker_imgpred_vse_m0.2_img2048_emb300_hid1024_reppen0.0_lr3e-04_linear_ann300k_drop0.0_clip1.0_",
            512 : "09.14_21.14._ranker_imgpred_vse_m0.2_img512_emb128_hid512_reppen0.0_lr3e-04_linear_ann500k_drop0.0_clip1.0_",
        },
    },

    "multi30k": {
        "vse":{
            "experiment": "180920_01_raw_vse_rc_flickr",
            2048 : "09.20_18.15._ranker_multi30k_imgpred_vse_m0.2_img2048_emb300_hid1024_lr3e-04_linear_ann30k_drop0.0_clip2.0_",
        },

        "mse":{
            "experiment": "180919_08_raw_mse_pretrained",
            2048 : "09.19_22.44._ranker_multi30k_imgpred_mse_m0.2_img2048_emb256_hid512_lr1e-03_linear_ann10k_drop0.4_clip2.0_",
        }
    }
}
