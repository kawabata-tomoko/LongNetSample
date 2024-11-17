这个仓库基于[torchscale](https://github.com/microsoft/torchscale/tree/main)进行修改：

- 删除了有关MOE、VisionEmbedding、MultiWay等内容
- 简化代码处理逻辑：
  - 将EncoderConfig、DecoderConfig等合并为LongnetConfig
  - 基于ESMTokenizer构造了LongNetTokenizer
  - 将稀疏注意力的支持合并到config设定中
- 使用Transformers库进行封装，针对语言模型特化
  - 提供LongNetPretrainedModel，对原始repo中分散的参数初始化进行简化合并
  - 提供三类BaseModel：EncoderOnly（LongNetEncoderLM）、DecoderOnly（LongNetDecoderLM）、Seq2SeqModel（LongNetEncoderDecoderLM）
  - 提供基于EncoderOnly和DecoderOnly的序列分类模型示例以及对应的预训练模型示例
  - 提供序列分类训练示例脚本[sample](./launcher_sars-substrain.py)

This repository is modified based on [torchscale](https://github.com/microsoft/torchscale/tree/main):

* Removed content related to MOE, VisionEmbedding, MultiWay, etc.
* Simplified code processing logic:
  * Merged EncoderConfig, DecoderConfig, etc., into LongnetConfig
  * Constructed LongNetTokenizer based on ESMTokenizer
  * Integrated support for sparse attention into the config settings
* Encapsulated using the Transformers library, specialized for language models:
  * Provided LongNetPretrainedModel to simplify and merge parameter initialization scattered in the original repo
  * Provided three types of BaseModel: EncoderOnly (LongNetEncoderLM), DecoderOnly (LongNetDecoderLM), Seq2SeqModel (LongNetEncoderDecoderLM)
  * Provided examples of sequence classification models based on EncoderOnly and DecoderOnly, along with corresponding pre-trained model examples
  * Provided a sequence classification training example script [sample](./launcher_sars-substrain.py)
