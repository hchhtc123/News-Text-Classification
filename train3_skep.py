import pandas as pd
import math
import numpy as np
import os
import collections
from functools import partial
import random
import time
import inspect
import importlib
from tqdm import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import IterableDataset
from paddle.utils.download import get_path_from_url
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import MapDataset
from paddle.dataset.common import md5file
from paddlenlp.datasets import DatasetBuilder
from paddlenlp.transformers import LinearDecayWithWarmup
import paddle.nn.functional as F

# 定义数据集对应文件及其文件存储格式
class NewsData(DatasetBuilder):
    SPLITS = {
        'train': 'train.csv',  # 训练集
        'dev': 'dev.csv',      # 验证集
    }

    def _get_data(self, mode, **kwargs):
        filename = self.SPLITS[mode]
        return filename

    def _read(self, filename):
        """读取数据"""
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")    # 以'\t'分隔各列
                if not head:
                    head = data
                else:
                    text_a, label = data
                    yield {"text_a": text_a, "label": label}  # 此次设置数据的格式为：text_a,label，可以根据具体情况进行修改

    def get_labels(self):
        return label_list   # 类别标签

# 定义数据集加载函数
def load_dataset(name=None,
                 data_files=None,
                 splits=None,
                 lazy=None,
                 **kwargs):
   
    reader_cls = NewsData  # 加载定义的数据集格式
    print(reader_cls)
    if not name:
        reader_instance = reader_cls(lazy=lazy, **kwargs)
    else:
        reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

    datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
    return datasets

# 定义数据加载和处理函数
def convert_example(example, tokenizer, max_seq_length=128, is_test=False):
    qtconcat = example["text_a"]
    encoded_inputs = tokenizer(text=qtconcat, max_seq_len=max_seq_length)  # tokenizer处理为模型可接受的格式 
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

# 定义数据加载函数dataloader
def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    # 训练数据集随机打乱，测试数据集不打乱
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

# 定义模型训练验证评估函数
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))  # 输出验证集上评估效果
    model.train()
    metric.reset()
    return accu  # 返回准确率

# 定义模型预测函数
def predict(model, data, tokenizer, label_map, batch_size=1):
    examples = []
    # 将输入数据（list格式）处理为模型可接受的格式
    for text in data:
        input_ids, segment_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=128,
            is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results  # 返回预测结果

# 定义对数据的预处理函数,处理为模型输入指定list格式
def preprocess_prediction_data(data):
    examples = []
    for text_a in data:
        examples.append({"text_a": text_a})
    return examples

# 将list格式的预测结果存储为txt文件，提交格式要求：每行一个类别
def write_results(labels, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        f.writelines("\n".join(labels))

if __name__ == '__main__':
    train = pd.read_table('./train.txt', sep='\t',header=None)  # 训练集
    dev = pd.read_table('./dev.txt', sep='\t',header=None)      # 验证集（官方已经划分的）
    test = pd.read_table('./test.txt', sep='\t',header=None)    # 测试集

    # 由于数据集存放时无列名，因此手动添加列名便于对数据进行更好处理
    train.columns = ["text_a",'label']
    dev.columns = ["text_a",'label']
    test.columns = ["text_a"]

    # 读取伪标签数据（将模型对无标签的测试集的预测结果加入到训练中去）
    newtest = pd.read_csv('/home/aistudio/work/newtest1.csv')
    # 拼接训练集和伪标签数据，通过加入伪标签数据，增大训练数据量提升模型泛化能力
    train = pd.concat([train,newtest],axis=0)

    # 保存处理后的数据集文件
    train.to_csv('train.csv', sep='\t', index=False)  # 保存训练集，格式为text_a,label，以\t分隔开
    dev.to_csv('dev.csv', sep='\t', index=False)      # 保存验证集，格式为text_a,label，以\t分隔开
    test.to_csv('test.csv', sep='\t', index=False)    # 保存测试集，格式为text_a，以\t分隔开

    # skep_ernie_1.0_large_ch模型
    # 指定模型名称，一键加载模型
    model = ppnlp.transformers.SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch", num_classes=14)
    # 同样地，通过指定模型名称一键加载对应的Tokenizer，用于处理文本数据，如切分token，转token_id等
    tokenizer = ppnlp.transformers.SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch")

    # 定义要进行分类的14个类别
    label_list=list(train.label.unique())

    # 加载训练和验证集
    train_ds, dev_ds = load_dataset(splits=["train", "dev"])

    # 参数设置：
    # 批处理大小，显存如若不足的话可以适当改小该值
    batch_size = 300
    # 文本序列最大截断长度，需要根据文本具体长度进行确定，最长不超过512。 通过文本长度分析可以看出文本长度最大为48，故此处设置为48
    max_seq_length = 48

    # 将数据处理成模型可读入的数据格式
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack()  # labels
    ): [data for data in fn(samples)]

    # 训练集迭代器
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    # 验证集迭代器
    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    # 定义训练配置参数：
    # 定义训练过程中的最大学习率
    learning_rate = 4e-5
    # 训练轮次
    epochs = 4
    # 学习率预热比例
    warmup_proportion = 0.1
    # 权重衰减系数，类似模型正则项策略，避免模型过拟合
    weight_decay = 0.0

    num_training_steps = len(train_data_loader) * epochs
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

    # AdamW优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    criterion = paddle.nn.loss.CrossEntropyLoss()  # 交叉熵损失函数
    metric = paddle.metric.Accuracy()              # accuracy评价指标

    # 固定随机种子便于结果的复现
    seed = 1024
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    save_dir = "checkpoint"
    if not  os.path.exists(save_dir):
        os.makedirs(save_dir)

    pre_accu=0
    accu=0
    global_step = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 10 == 0 :
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        # 每轮结束对验证集进行评估
        accu = evaluate(model, criterion, metric, dev_data_loader)
        print(accu)
        if accu > pre_accu:
            # 保存较上一轮效果更优的模型参数
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')  # 保存模型参数
            paddle.save(model.state_dict(), save_param_path)
            pre_accu=accu
    tokenizer.save_pretrained(save_dir)

    params_path = 'checkpoint/model_state.pdparams'
    if params_path and os.path.isfile(params_path):
        # 加载模型参数
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % params_path)

    # 测试最优模型参数在验证集上的分数
    evaluate(model, criterion, metric, dev_data_loader)

    # 定义要进行分类的类别
    label_list=list(train.label.unique())
    label_map = { 
        idx: label_text for idx, label_text in enumerate(label_list)
    }
    print(label_map)

    # 读取要进行预测的测试集文件
    test = pd.read_csv('./test.csv',sep='\t')  

    # 对测试集数据进行格式处理
    data1 = list(test.text_a)
    examples = preprocess_prediction_data(data1)

    # 对测试集进行预测
    results = predict(model, examples, tokenizer, label_map, batch_size=16)   
    write_results(results, "./train3_skep_result.txt")