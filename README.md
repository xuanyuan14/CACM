## A Pytorch Implementation of Context-Aware Click Model (CACM)

### Introduction

This codebase contains source-code of the Pytorch-based implementation of our WSDM2020 paper.
  - [WSDM 2020] [A Context-Aware Click Model for Web Search](https://dl.acm.org/doi/10.1145/3336191.3371819)

### Requirements

* python 3.7
* pytorch 1.4.0
* [tensorboardX](https://pypi.python.org/pypi/tensorboardX)
* [tqdm](https://pypi.org/project/tqdm/)
* [prettytable](https://pypi.org/project/PrettyTable/)


### Available Dataset

We are delighted to share the public session dataset we used to run our experiments. This Chinese-centric TianGong-ST dataset is provided to support researches in a wide range of session-level Information Retrieval (IR) tasks. It consists of 147,155 refined Web search sessions, 40,596 unique queries, 297,597 Web pages, six kinds of weak relevance labels assessed by click models, and also a subset of 2,000 sessions with 5-level human relevance labels for documents of the last queries in them. In our experiments, the dataset is splitted into training, validating and testing set with a ratio of 8:1:1. To ensure proper evaluation, we filter a session in the validating and testing set if it contains queries which do not appear in the training set. We also include all the annotated sessions in the testing set to facilitate the evaluation of relevance estimation. Some specifics of the dataset can be found as follows:

| Attribute           |   Train |  Dev   |   Test |
| :---: | :--: | :---: | :---: |
| Sessions            |  117431 | 13154  |  26570 |
| Queries             | 35903 | 9373 | 11391 |
| Avg Session Len     |    2.4099 |  2.4012  |   2.4986 |

This dataset is now available at [here](http://www.thuir.cn/tiangong-st/).


### Data Preparation
- We provide data pre-processing codes to generate all the needed input files from TianGong-ST dataset. All the generated files should be put under the ```./data``` directory.
    - [TianGong-ST-CACM.py](TianGong-ST-CACM.py)
- You should generate your embedding files through the [node2vec](https://github.com/snap-stanford/snap/tree/master/examples/node2vec) tool and put it under the ```./data``` directory.
- Sample session files are available under the ```./data``` directory. The format of sample session files is as follows:
    - each line: ```[<query sequence>]<tab>[<previous interaction>]<tab>[<document info>]<tab><clicked>```
    - query sequence: qids 
    - interaction sequence: uid, rank, vid, clicked
    - document info: uid, rank, vid
    - clicked: 0 or 1


### Quick Start

To do data pre-processing, run the following command:

```c
python TianGong-ST-CACM.py \
--clean_xml --dict_list --txt --node2vec --human_label_txt_for_CACM \
--dataset sogousessiontrack2020.xml \
--input ../dataset/TianGong-ST/data/ \
--output ./data \
```

To train CACM model on a small data sample, run the following command:

```c
python -u run.py --train --optim adam --eval_freq 5 --check_point 5 \
--learning_rate 0.001 --weight_decay 1e-5 --dropout_rate 0.5 --batch_size 1 \
--num_steps 200000 --embed_size 64 --hidden_size 256 --patience 5 \
--use_knowledge True --use_state_attention True --use_knowledge_attention True \
--train_dirs ./data/train_per_session.txt \
--dev_dirs ./data/dev_per_session.txt \
--test_dirs ./data/test_per_session.txt \
--label_dirs ./data/human_label_for_CACM.txt \
--data_dir ./data/ \
--model_dir ./outputs/models/ \
--result_dir ./outputs/results/ \
--summary_dir ./outputs/summary/ \
--log_dir ./outputs/log/ 
```

**NOTE**: due to the difference of the session size, the batch_size can only be set to one. We will accumulate the gradient and update at every 32 iterations to simulate the logical batch_size of 32. Check out at line 151 in [model.py](model.py)


### Citation

If you find the resources in this repo useful, please cite our work.

```
@inproceedings{chen2020Context,
  title={A Context-Aware Click Model for Web Search},
  author={Chen, Jia and Mao, Jiaxin and Liu, Yiqun and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the Thirteenth ACM International Conference on Web Search and Data Mining},
  year={2020},
  organization={ACM}
}
```

### Acknowledgement

Thanks to [Jianghao Lin](https://chiangel.github.io/) for:

- Upgrade codes for python3
- Data preprecessing codes for TianGong-ST
- Evaluation codes for LL/PPL/NDCG