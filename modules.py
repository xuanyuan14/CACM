#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ref: A Context-Aware Click Model for Web Search
@author: Jia Chen, Jiaxin Mao, Yiqun Liu, Min Zhang, Shaoping Ma
@desc: The implementation of each module in CACM
'''
import torch
import logging
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()


# query context encoder
class KnowledgeEncoder(nn.Module):  
    def __init__(self, args, input_size, n_layers=1):
        super(KnowledgeEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = args.hidden_size
        self.use_knowledge = args.use_knowledge
        self.embed_size = args.embed_size

        # if use pre-trained embeddings, then there is no need to use embedding layers
        self.embedding = nn.Embedding(input_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, batch_first=True)

    def forward(self, input, hidden, data, query_len):
        node_emb = data.node_emb
        qid_nid = data.qid_nid
        if self.use_knowledge:
            try:  # load the embeddings
                output = input.data.cpu().numpy().tolist()
                output = node_emb[qid_nid[str(output).decode('utf-8')]]
                output = Variable(torch.from_numpy(np.array(output, dtype=np.float32)))
                if use_cuda:
                    output = output.cuda()
            except:
                embedded = self.embedding(input)
                output = embedded
        else:
            embedded = self.embedding(input)
            output = embedded
        output = output.view(1, query_len, -1)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output[-1], hidden[-1]

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# click context encoder，encode all user interactions within a session
class StateEncoder(nn.Module):
    def __init__(self, args, url_size, vtype_size, rank_size=11, n_layers=1):
        super(StateEncoder, self).__init__()
        self.args = args
        self.n_layers = n_layers
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.use_knowledge = args.use_knowledge
        self.encode_gru_num_layer = 1

        self.url_size = url_size
        self.rank_size = rank_size
        self.vtype_size = vtype_size

        self.url_embedding = nn.Embedding(url_size, self.embed_size)
        self.rank_embedding = nn.Embedding(rank_size, 4)
        self.vtype_embedding = nn.Embedding(vtype_size, 8)
        self.action_embedding = nn.Embedding(2, 4)

        self.gru = nn.GRU(self.embed_size + 16, self.hidden_size,
                          batch_first=True, dropout=self.dropout_rate, num_layers=self.encode_gru_num_layer)

    def forward(self, urls, ranks, vtypes, actions, hidden, data):
        uid_nid = data.uid_nid
        node_emb = data.node_emb
        if self.use_knowledge:
            batch_embeds = []
            for url_batch in urls:
                batch_embed = []
                for url in url_batch:
                    try:
                        this_embed = url.data.cpu().numpy().tolist()
                        this_embed = node_emb[uid_nid[str(this_embed).decode('utf-8')]]
                        this_embed = Variable(torch.from_numpy(np.array(this_embed, dtype=np.float32)))
                        if use_cuda:
                            this_embed = this_embed.cuda()
                    except:
                        this_embed = self.url_embedding(url)
                    batch_embed.append(this_embed)
                batch_embed = torch.stack(tuple(batch_embed), dim=0)
                batch_embeds.append(batch_embed)
            url_embed = torch.stack(tuple(batch_embeds), dim=0)
        else:
            url_embed = self.url_embedding(urls)  # batch_size, session_doc_num, embed_size
        rank_embed = self.rank_embedding(ranks)  # batch_size, session_doc_num, 4
        vtype_embed = self.vtype_embedding(vtypes)  # batch_size, session_doc_num, 8
        action_embed = self.action_embedding(actions)  # batch_size, session_doc_num, 4

        gru_input = torch.cat((url_embed, rank_embed, vtype_embed, action_embed), dim=2)
        output = gru_input
        for i in range(self.n_layers):
            output, hidden = self.gru(gru_input, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# document encoder, encode one document each time
class DocumentEncoder(nn.Module):
    def __init__(self, args, url_size, vtype_size, rank_size=11, n_layers=1):
        super(DocumentEncoder, self).__init__()
        self.args = args
        self.n_layers = n_layers
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.use_knowledge = args.use_knowledge
        self.encode_gru_num_layer = 1

        self.url_size = url_size
        self.rank_size = rank_size
        self.vtype_size = vtype_size

        self.url_embedding = nn.Embedding(url_size, self.embed_size)
        self.rank_embedding = nn.Embedding(rank_size, 4)
        self.vtype_embedding = nn.Embedding(vtype_size, 8)
        self.qcnt_embedding = nn.Embedding(11, 4)
        self.output_linear = nn.Linear(self.embed_size+16, self.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, urls, ranks, vtypes, q_iter, data):
        node_emb = data.node_emb
        uid_nid = data.uid_nid
        if self.use_knowledge:
            batch_embeds = []
            for url_batch in urls:
                batch_embed = []
                for url in url_batch:
                    try:
                        this_embed = url.data.cpu().numpy().tolist()
                        this_embed = node_emb[uid_nid[str(this_embed).decode('utf-8')]]
                        this_embed = Variable(torch.from_numpy(np.array(this_embed, dtype=np.float32)))
                        # url_embed = url_embed.view(1, 1, -1)
                        if use_cuda:
                            this_embed = this_embed.cuda()
                    except:
                        this_embed = self.url_embedding(url)
                    batch_embed.append(this_embed)
                batch_embed = torch.stack(tuple(batch_embed), dim=0)
                batch_embeds.append(batch_embed)
            url_embed = torch.stack(tuple(batch_embeds), dim=0)
        else:
            url_embed = self.url_embedding(urls)  # batch_size, session_doc_num, embed_size
        rank_embed = self.rank_embedding(ranks)  # batch_size, session_doc_num, 4
        vtype_embed = self.vtype_embedding(vtypes)  # batch_size, session_doc_num, 8
        qcnt_embed = self.qcnt_embedding(q_iter)  # batch_size, session_doc_num, 4

        doc_embed = torch.cat((url_embed, rank_embed, vtype_embed, qcnt_embed), dim=2)
        doc_embed = self.tanh(self.output_linear(doc_embed))
        return doc_embed


# relevance estimator
class RelevanceEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(RelevanceEstimator, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.out1 = nn.Linear(input_size, hidden_size / 2)
        self.out2 = nn.Linear(hidden_size / 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, batch_size):
        output = self.tanh(self.out1(input))
        output = self.sigmoid(self.out2(output)).view(batch_size, -1, 1)
        return output


# examination predictor
class ExamPredictor(nn.Module):
    def __init__(self, args, vtype_size, rank_size=11):
        super(ExamPredictor, self).__init__()
        self.args = args
        self.logger = logging.getLogger("CACM")
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.encode_gru_num_layer = 1
        self.vtype_size = vtype_size

        self.rank_embedding = nn.Embedding(rank_size, 4)
        self.vtype_embedding = nn.Embedding(vtype_size, 8)
        self.action_embedding = nn.Embedding(2, 4)

        self.gru = nn.GRU(16, self.hidden_size,
                          batch_first=True, dropout=self.dropout_rate, num_layers=self.encode_gru_num_layer)

        self.output_linear = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vtype, action, rank, init_gru_state):

        rank_embed = self.rank_embedding(rank)
        vtype_embed = self.vtype_embedding(vtype)
        action_embed = self.action_embedding(action)

        gru_input = torch.cat((rank_embed, vtype_embed, action_embed), dim=2)
        outputs, hidden = self.gru(gru_input, init_gru_state)
        exams = self.sigmoid(self.output_linear(outputs))
        return exams

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result