### 对MT5模型进行中文任务的精简
---
总体思路采用[苏剑林老师的思路][1]的思路，但是苏老师的代码是keras的，然而现在大家更多的都是用Pytorch写代码。我找到了这个[Github简化链接][2]但是这个代码写的不详细而且有点错误。于是我把代码改正以后上传到了[Github][3]并在这儿写下详细使用方法。

###1. 生成需保留词表
复制原模型文件夹中的spiece.model和config.json文件至新文件夹mt5-large-simplify并改名为spiece_cn.model，并将result.json复制到spm_simplify.py文件夹。只需要修改spm_simplify.py代码下面两行。然后执行spm_simplify.py代码对spiece.model进行tokenizer的精简，保留前259和后100个标记，中文标记保留result.json文件中的出现频率最高的3w多个token，脚本会生成sentencepiece_cn_keep_tokens.json文件，里面包含了保留的所有原始mt5token的索引。
```
#spm_simplify.py
#根据实际情况下述两个路径
old_model = '../model/mt5-large-simplify/spiece.model'
new_model = '../model/mt5-large-simplify/spiece_cn.model'
```

###2. 对模型参数进行简化
按照下面指示修改config.yaml（注意如下注释仅辅助理解在文件中不用加入）,根据步骤1生成的sentencepiece_cn_keep_tokens.json文件使用simplify.py对pytorch_model.bin文件进行精简，将shared.weight、encoder.embed_tokens、decoder.embed_tokens和lm_head.embed_tokens参数根据json文件中的索引进行保留并保存。
```
#config.yaml
#原模型路径
source_path: ../model/mt5-large
#新模型路径
target_path: ../model/mt5-large-simplify
```
###3. 对模型保留参数
按照如下指示通过调用verify(config)函数求取保留模型参数大小（一般情况下为32598）。
```
#simplify.py
#简化代码完成以后将下面simp代码注释
simp(config)
#并取消下面verify代码的注释
#verify(config)
```
###4. 修改模型配置文件
修改mt5-large-simplify文件夹中的config.json的vocab_size为上面求得的模型参数32598



  [1]: https://spaces.ac.cn/archives/7867/comment-page-1
  [2]: https://github.com/yangyubuaa/mt5_simplify
  [3]: https://github.com/Hi-archers/MT5_chinese_simplify
