# 飞桨常规赛：中文新闻文本标题分类 9月第1名方案

# 一.赛题介绍：

本次比赛赛题为新闻标题文本分类 ，选手需要根据提供的新闻标题文本和类别标签训练一个新闻分类模型，然后对测试集的新闻标题文本进行分类，评价指标上使用Accuracy = 分类正确数量 / 需要分类总数量。同时本次参赛选手需使用飞桨框架和飞桨文本领域核心开发库PaddleNLP，PaddleNLP具备简洁易用的文本领域全流程API、多场景的应用示例、非常丰富的预训练模型，深度适配飞桨框架2.x版本。

比赛传送门：https://aistudio.baidu.com/aistudio/competition/detail/107/0/introduction

# 二.项目简介：

飞桨常规赛：中文新闻文本标题分类9月第1名方案，分数0.9+，基于PaddleNLP通过预训练模型的微调完成新闻14分类模型的训练与优化，并利用训练好的模型对测试数据进行预测并生成提交结果文件。最后主要通过伪标签和结果融合trick对效果进行进一步的提升。

# 三.AI Studio项目地址：

https://aistudio.baidu.com/aistudio/projectdetail/2345384

# 四.运行说明：

本地运行的话需下载本项目并根据提供的“运行说明.txt”文件进行操作完成本地训练。也可以进入AI Studio项目地址进行运行。

# 五.进一步优化方向：

a.针对训练存在的过拟合问题，可以考虑在划分训练和验证集时进行下数据均衡。同时可以尝试通过回译+同义词替换+相似句替换等的数据增强方法对训练数据进行扩增提升模型泛化能力。

b.伪标签技巧个人采用的是取多模型预测相同的部分，也可以尝试在预测输出结果时同时输出预测结果的置信度，取结果中置信度较高的作为伪标签加入训练，该方法在数据标注中较为常见。

c.目前在单模上分数较低，可以考虑通过调参、优化模型等思路进一步提高单模的效果。

d.感兴趣的可以研究下prompt和PET,对这块感兴趣的可以参考下 格局打开，带你解锁 prompt 的花式用法（https://mp.weixin.qq.com/s/RFGbX1Np5KkVfij8d6AczQ）
这块也是目前正在学习和尝试实践的地方。

e.更多技巧主要可以到天池或kaggle等数据科学平台或Github搜索，通过学习类似文本分类比赛的Top分享去进行学习和实践，以赛促学。

关于PaddleNLP的使用：建议多看官方最新文档：https://paddlenlp.readthedocs.io/zh/latest/get_started/quick_start.html

PaddleNLP的github地址：https://github.com/PaddlePaddle/PaddleNLP 有问题的话可以在github上提issue，会有专人回答。
