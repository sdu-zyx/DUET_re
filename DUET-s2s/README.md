##Dataset ML-1M
### data
movielens-1m from RecBole, released by RUC https://github.com/RUCAIBox/RecBole
### data process
seq2seq training strategy
```shell script
实验数据集划分:将数据集划分成两部分,分别用于pretrain和hyper network train

用于pretrain, pre_train.pkl, pre_valid.pkl
利用前40%的数据预训练序列编码器,并使用验证集找最优epoch,不设测试集


用于hyper network训练阶段,训练序列编码器和hyper network,其中分类器依靠hyper network生成(生成分类器参数)
对应hyper_train.pkl,hyper_valid.pkl,hyper_test.pkl
```

```shell script
base model end2end 训练全部的数据样例
对应base_train.pkl, base_valid.pkl, hyper_test.pkl
```

```shell script
ml-1m.txt 保存所有的用户序列
ml-1m_sample.txt testing时负采样100
```

##Running Example
可以结合不同的数据处理、和训练策略
###pretrain
```shell script
python runner_pretrain.py
```
###hyper network

```shell script
python runner_hyper.py

113行代码pretrained_path = os.path.join(args.output_dir, f'DUET-pre-{args.data_name}.pt')
加载预训练模型,不加载预训练模型时用base_train.pkl, base_valid.pkl, hyper_test.pkl可训练全量数据的网络,效果较好
```

###base finetune
```shell script
python runner_base_ft.py

加载预训练模型,可以不改变预训练的分类器结构继续训练作为对比
```

###base 
```shell script
python runner_base.py

end2end, 使用所有的数据样例训练模型
```

