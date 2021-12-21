import yaml
import torch
import os
import json
from collections import OrderedDict

from mt5_model.transformers_pkg.models.mt5 import MT5Model

def simp(config):
    source_path = config["source_path"]
    target_path = config["target_path"]

        # 加载原始参数到source_state_orderdict
    source_state_orderdict = torch.load(os.path.join(source_path, "pytorch_model.bin"))

    # print(source_state_orderdict)

    # 得到shared参数（即embedding参数，encoder和decoder共享）
    # (250112, 768) 我们可以计算出shared层占用的参数量为250112 * 768 = 192M
    # 占用的内存为768MB左右，所以需要进行精简
    # 读取id，假设id为[0, 2]
    with open("sentencepiece_cn_keep_tokens.json", "r", encoding="utf-8") as r:
        id = json.load(r)
    #print(id)

    #print(simplified.shape)
    new_embedding_params = OrderedDict()

    shared = source_state_orderdict["shared.weight"]
    simplified = torch.index_select(shared, 0, torch.tensor(id))
    new_embedding_params["shared.weight"] = simplified

    shared = source_state_orderdict["encoder.embed_tokens.weight"]
    simplified = torch.index_select(shared, 0, torch.tensor(id))
    new_embedding_params["encoder.embed_tokens.weight"] = simplified

    shared = source_state_orderdict["decoder.embed_tokens.weight"]
    simplified = torch.index_select(shared, 0, torch.tensor(id))
    new_embedding_params["decoder.embed_tokens.weight"] = simplified


    shared = source_state_orderdict["lm_head.weight"]
    simplified = torch.index_select(shared, 0, torch.tensor(id))
    new_embedding_params["lm_head.weight"] = simplified

    # 更新source_state_dict参数
    source_state_orderdict.update(new_embedding_params)

    #print(source_state_orderdict, target_path)

    torch.save(source_state_orderdict, target_path + '/pytorch_model.bin')
    # with open(target_path, "w") as w:
    #     json.dump(source_state_orderdict, w)


def verify(config):
    m = MT5Model.from_pretrained(config["target_path"])
    for name, params in m.named_parameters():
        print(params.shape)
        break


if __name__ == '__main__':
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
        
    #simplify.py
    #简化代码完成以后将下面simp代码注释
    simp(config)
    #并取消下面verify代码的注释
    #verify(config)
