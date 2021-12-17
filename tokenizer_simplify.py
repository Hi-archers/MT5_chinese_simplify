import sentencepiece as spm
import torch
import os

if __name__ == '__main__':
    s1 = spm.SentencePieceProcessor(model_file='../model/mt5-large/spiece.model')
    s2 = spm.SentencePieceProcessor(model_file="../model/mt5-large-simplify/spiece_cn.model")
    print(s1.Encode("我"))
    print(s2.Encode("我"))

    source_state_orderdict = torch.load(os.path.join(  "../model/mt5-large", "pytorch_model.bin"))
    simplify_state_orderdict = torch.load(os.path.join("../model/mt5-large-simplify", "pytorch_model.bin"))
    print(simplify_state_orderdict.__sizeof__())
    s1_e = source_state_orderdict["shared.weight"]
    s2_e = simplify_state_orderdict["shared.weight"]

    print(s1_e[3003] == s2_e[1182])
    print(s2_e[1182])

    s1_e = source_state_orderdict["encoder.embed_tokens.weight"]
    s2_e = simplify_state_orderdict["encoder.embed_tokens.weight"]
    print(s1_e[3003] == s2_e[1182])
    print(s2_e[1182])

    s1_e = source_state_orderdict["decoder.embed_tokens.weight"]
    s2_e = simplify_state_orderdict["decoder.embed_tokens.weight"]
    print(s1_e[3003] == s2_e[1182])
    print(s2_e[1182])


    s1_e = source_state_orderdict["lm_head.weight"]
    s2_e = simplify_state_orderdict["lm_head.weight"]
    print(s1_e[3003] == s2_e[1182])
    print(s2_e[1182])
