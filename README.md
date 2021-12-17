### 对Mt5模型进行中文任务的精简
---
##### 精简目标
1. tokenizer
2. embedding参数
##### 精简步骤
1. 使用spm_simplify.py脚本对sentencepiece.model进行tokenizer的精简，保留前259和后100个标记，中文标记保留result.json文件中的出现频率最高的3w多个token，脚本会生成sentencepiece_cn_keep_tokens.json文件，里面包含了保留的所有原始mt5token的索引。
2. 根据步骤1生成的json文件使用simplify对pytorch_model.bin文件进行精简，将shared.weight参数根据json文件中的索引进行保留并保存。
3. 对config.json中的词表大小进行修改，修改为精简后的词表大小。
##### 测试步骤
测试脚本为test.py以及tokenizer_simplify.py，主要是测试精简前后的token的对应信息，以及看精简前后相同的token对应的embedding是否相同，如果相同，则精简成功。
