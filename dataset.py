#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ref: A Context-Aware Click Model for Web Search
@author: Jia Chen, Jiaxin Mao, Yiqun Liu, Min Zhang, Shaoping Ma
@desc: Loading the dataset
'''
import json
import logging
import numpy as np


class Dataset(object):
    def __init__(self, args, train_dirs=[], dev_dirs=[], test_dirs=[], isRank=False):
        self.logger = logging.getLogger("CACM")
        self.max_d_num = args.max_d_num
        self.gpu_num = args.gpu_num
        self.args = args
        self.num_train_files = args.num_train_files
        self.num_dev_files = args.num_dev_files
        self.num_test_files = args.num_test_files
        self.embed_size = args.embed_size

        # load the pre-trained embeddings if use knowledge
        self.node_emb = {}
        self.qid_nid = {}
        self.uid_nid = {}
        if args.use_knowledge:
            knowledge_type = args.knowledge_type
            knowledge_dir = './graph/%s/edge_%s.emb' % (knowledge_type, self.embed_size)
            with open(knowledge_dir, 'r') as fp:
                fc = True
                for line in fp:
                    if fc:
                        _, n_emb = map(int, line.strip().split())
                        assert (n_emb == args.embed_size)
                        fc = False
                    else:
                        data = line.strip().split()
                        assert (len(data) == args.embed_size + 1)
                        self.node_emb[int(data[0])] = [float(x) for x in data[1:]]

            # load qid_nid, uid_nid
            with open('./data/dict/qid_nid.json') as f1:
                self.qid_nid = json.load(f1)
            with open('./data/dict/uid_nid.json') as f2:
                self.uid_nid = json.load(f2)

        self.train_set, self.dev_set, self.test_set = [], [], []
        if isRank:
            if test_dirs:
                for test_dir in test_dirs:
                    self.test_set += self.load_dataset_rank(test_dir, num=self.num_test_files, mode='test')
                self.logger.info('Test set size: {} sessions.'.format(len(self.test_set)))
        else:
            if train_dirs:
                for train_dir in train_dirs:
                    self.train_set += self.load_dataset(train_dir, num=self.num_train_files, mode='train')
                self.logger.info('Train set size: {} sessions.'.format(len(self.train_set)))
            if dev_dirs:
                for dev_dir in dev_dirs:
                    self.dev_set += self.load_dataset(dev_dir, num=self.num_dev_files, mode='dev')
                self.logger.info('Dev set size: {} sessions.'.format(len(self.dev_set)))

    def load_dataset(self, data_path, num, mode):
        data_set = []
        files = [data_path]
        if num > 0:
            files = files[0:num]

        sess_id = 1
        for dir in files:
            fn = open(dir, 'r')
            sess = fn.read().strip().split('\n\n')
            for s in sess:
                knowledge_qs, interactions, doc_infos, exams, clicks = [], [], [], [], []
                lines = s.strip().split('\n')
                dcnt = 0
                for line in lines:
                    attr = line.strip().split('\t')

                    this_knowledge_qs = json.loads(attr[0])
                    qlen = len(this_knowledge_qs)
                    # padding
                    if qlen < 10:
                        for i in range(10 - qlen):
                            this_knowledge_qs.append(0)
                    previous_interaction = json.loads(attr[1])
                    if len(previous_interaction) == 0:
                        previous_interaction = [0, 0, 0, 0]

                    this_doc_info = json.loads(attr[2])
                    qcnt = dcnt / 10 + 1
                    this_doc_info.append(qcnt)
                    this_click = int(attr[3])

                    knowledge_qs.append(this_knowledge_qs)
                    interactions.append(previous_interaction)
                    doc_infos.append(this_doc_info)
                    clicks.append(this_click)

                    # exam
                    if dcnt % 10 == 0:
                        exams.append([0, 0, 0, 0])
                    else:
                        exams.append(previous_interaction)

                    dcnt += 1

                data_set.append({'knowledge_qs': knowledge_qs,
                                 'interactions': interactions,
                                 'doc_infos': doc_infos,
                                 'clicks': clicks,
                                 'exams': exams,
                                 'sess_id': sess_id})
                sess_id += 1
        return data_set

    def load_dataset_rank(self, data_path, num, mode):
        data_set = []
        files = [data_path]
        if num > 0:
            files = files[0:num]

        sess_id = 1
        for dir in files:
            fn = open(dir, 'r')
            sess = fn.read().strip().split('\n\n')
            for s in sess:
                knowledge_qs, interactions, doc_infos, exams, clicks = [], [], [], [], []
                lines = s.strip().split('\n')
                dcnt = 0
                for line in lines:
                    attr = line.strip().split('\t')

                    this_knowledge_qs = json.loads(attr[0])
                    qlen = len(this_knowledge_qs)
                    # padding
                    if qlen < 10:
                        for i in range(10 - qlen):
                            this_knowledge_qs.append(0)
                    previous_interaction = json.loads(attr[1])
                    if len(previous_interaction) == 0:
                        previous_interaction = [0, 0, 0, 0]

                    this_doc_info = json.loads(attr[2])
                    qcnt = dcnt / 10 + 1
                    this_doc_info.append(qcnt)
                    this_click = int(attr[3])

                    knowledge_qs.append(this_knowledge_qs)
                    interactions.append(previous_interaction)
                    doc_infos.append(this_doc_info)
                    clicks.append(this_click)

                    # exam
                    if dcnt % 10 == 0:
                        exams.append([0, 0, 0, 0])
                    else:
                        exams.append(previous_interaction)

                    dcnt += 1

                data_set.append({'knowledge_qs': knowledge_qs,
                                 'interactions': interactions,
                                 'doc_infos': doc_infos,
                                 'clicks': clicks,
                                 'exams': exams,
                                 'sess_id': sess_id})
                sess_id += 1
        return data_set

    def _one_mini_batch(self, data, indices):
        batch_data = {'raw_data': [data[i] for i in indices],
                      'knowledge_qs': [],
                      'interactions': [],
                      'doc_infos': [],
                      'clicks': [],
                      'exams': []}
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['knowledge_qs'].append(sample['knowledge_qs'])
            batch_data['interactions'].append(sample['interactions'])
            batch_data['doc_infos'].append(sample['doc_infos'])
            batch_data['clicks'].append(sample['clicks'])
            batch_data['exams'].append(sample['exams'])
        return batch_data

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)

        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()

        indices += indices[:(self.gpu_num - data_size % self.gpu_num) % self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices)
