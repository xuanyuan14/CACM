## A Pytorch Implementation of Context-Aware Click Model (CACM)

### Introduction

This codebase contains source-code of the Pytorch-based implementation of our WSDM2020 paper.
  - [WSDM 2020] [A Context-Aware Click Model for Web Search](https://dl.acm.org/doi/10.1145/3336191.3371819)

### Requirements

* python 2.7.14
* pytorch >= 1.0.1
* [tensorboardX](https://pypi.python.org/pypi/tensorboardX)
* [tqdm](https://pypi.org/project/tqdm/)
* [prettytable](https://pypi.org/project/PrettyTable/)


### Data Preparation
- You should generate your embedding files through the [node2vec](https://github.com/snap-stanford/snap/tree/master/examples/node2vec) tool and put it under the ```./data/graph``` directory.
- You should create files which map queries and documents into node identifiers, rename them as ```qid_nid.json```and  ```uid_nid.json``` , and put them under the ```./data/dict``` directory.
- Sample session files are available under the ```./data``` directory. The format of sample session files is as follows:

* each line: [<query sequence>]<tab>[<previous interaction>]<tab>[<document info>]<tab><clicked>
* query sequence: qids 
* interaction sequence: uid, rank, vid, clicked
* document info: uid, rank, vid
* clicked: 0 or 1


### Available Dataset

We are delighted to share the public session dataset we used to run our experiments. This Chinese-centric TianGong-ST dataset is provided to support researches in a wide range of session-level Information Retrieval (IR) tasks. It consists of 147,155 refined Web search sessions, 40,596 unique queries, 297,597 Web pages, six kinds of weak relevance labels assessed by click models, and also a subset of 2,000 sessions with 5-level human relevance labels for documents of the last queries in them. In our experiments, the dataset is splitted into training, validating and testing set with a ratio of 8:1:1. To ensure proper evaluation, we filter a session in the validating and testing set if it contains queries which do not appear in the training set. We also include all the annotated sessions in the testing set to facilitate the evaluation of relevance estimation. Some specifics of the dataset can be found as follows:

| Attribute           |   Train |  Dev   |   Test |
| :--- | ---: | ---: | ---: |
| Sessions            |  117431 | 13154  |  26570 |
| Queries             | 35903 | 9373 | 11391 |
| Avg Session Len     |    2.4099 |  2.4012  |   2.4986 |

This dataset is now available at [here](http://www.thuir.cn/tiangong-st/).


### Quick Start

For example, to train CACM model on a small data sample, run the following command.

```
python -u run.py --train --train_dirs ../data/sample_train_sess.txt --dev_dirs ../data/sample_test_sess.txt --num_train_files 1 --num_dev_files 1 --optim adagrad --eval_freq 5 --check_point 5 --learning_rate 0.01 --weight_decay 1e-5 --dropout_rate 0.2 --batch_size 1 --num_steps 20 --embed_size 128 --hidden_size 256 --max_d_num 10 --model_dir ../data/models_sample --result_dir ../data/results_sample --summary_dir ../data/summary_sample --patience 5 --use_state_attention True --use_knowledge_attention True
```

Note that due to the difference of the session size, the batch_size can only be set to one.


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
