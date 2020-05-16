#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ref: A Context-Aware Click Model for Web Search
@author: Anonymous Author(s)
@desc: The implementation of CACM
'''
import logging
import torch
from torch import nn
from modules import KnowledgeEncoder, StateEncoder, DocumentEncoder, RelevanceEstimator, ExamPredictor

MINF = 1e-30
use_cuda = torch.cuda.is_available()


class CACMN(nn.Module):
    def __init__(self, args, query_size, url_size, vtype_size, n_layers=1):
        super(CACMN, self).__init__()
        self.n_layers = n_layers
        self.args = args
        self.knowledge_hidden_size = args.hidden_size
        self.state_hidden_size = args.hidden_size
        self.document_hidden_size = args.hidden_size
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.embed_size = args.embed_size

        self.softmax1 = torch.nn.Softmax(dim=0)
        self.softmax2 = torch.nn.Softmax(dim=1)
        self.logger = logging.getLogger("CACM")
        self.query_size = query_size
        self.url_size = url_size
        self.vtype_size = vtype_size
        self.dropout_rate = args.dropout_rate
        self.encode_gru_num_layer = 1
        self.use_knowledge = args.use_knowledge
        self.use_knowledge_attention = args.use_knowledge_attention
        self.use_state_attention = args.use_state_attention

        # whether use pre-trained embeddings
        if args.use_knowledge:
            self.knowledge_embedding_size = args.embed_size
        else:
            self.knowledge_embedding_size = query_size

        # context-aware relevance estimator
        self.knowledge_encoder = KnowledgeEncoder(self.args, self.query_size)
        self.state_encoder = StateEncoder(self.args, self.url_size, self.vtype_size)
        self.document_encoder = DocumentEncoder(self.args, self.url_size, self.vtype_size)
        self.relevance_estimator = RelevanceEstimator(self.args.hidden_size * 3, args.hidden_size)

        # examination predictor
        self.examination_predictor = ExamPredictor(self.args, self.vtype_size)

        # set the combination function of relevance and examination
        if self.args.combine == 'exp_mul' or self.args.combine == 'exp_sigmoid_log':
            self.lamda = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.mu = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

            # initialization
            self.lamda.data.fill_(1.0)
            self.mu.data.fill_(1.0)

        elif self.args.combine == 'linear':
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

            self.alpha.data.fill_(0.5)
            self.beta.data.fill_(0.5)

        elif self.args.combine == 'nonlinear':
            self.w11 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w12 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w21 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w22 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w31 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w32 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.sigmoid = nn.Sigmoid()

            self.w11.data.fill_(0.5)
            self.w12.data.fill_(0.5)
            self.w21.data.fill_(0.5)
            self.w22.data.fill_(0.5)
            self.w31.data.fill_(0.5)
            self.w32.data.fill_(0.5)

    def get_clicks(self, relevances, exams):
        clicks = []
        combine = self.args.combine
        if combine == 'mul':
            clicks = torch.mul(relevances, exams)
        elif combine == 'exp_mul':
            clicks = torch.mul(torch.pow(relevances, self.lamda), torch.pow(exams, self.mu))
        elif combine == 'linear':
            clicks = torch.add(torch.mul(relevances, self.alpha), torch.mul(exams, self.beta))
        elif combine == 'nonlinear':  # 2-layer
            out1 = self.sigmoid(torch.add(torch.mul(relevances, self.w11), torch.mul(exams, self.w12)))
            out2 = self.sigmoid(torch.add(torch.mul(relevances, self.w21), torch.mul(exams, self.w22)))
            clicks = self.sigmoid(torch.add(torch.mul(out1, self.w31), torch.mul(out2, self.w32)))
        elif combine == 'sigmoid_log':
            clicks = 4 * torch.div(torch.mul(relevances, exams),
                                   torch.mul(torch.add(relevances, 1), torch.add(exams, 1)))

        return clicks

    # inputs include: knowledge, interaction, document
    def forward(self, knowledge_variable, interaction_variable, document_variable, examination_context, data):
        # every variable correspond to a query-doc pair, which is to be predicted
        # forward one query session at a time

        # knowledge encoding
        knowledge_input_variable = knowledge_variable
        knowledge_input_variable = knowledge_input_variable.cuda() if use_cuda else knowledge_input_variable

        knowledge_output_list = []
        for batch_idx, batch_knowledge in enumerate(knowledge_input_variable):
            batch_knowledge_output = []
            for sess_pos_idx, knowledge in enumerate(batch_knowledge):
                query_idx = sess_pos_idx / 10 + 1
                knowledge_hidden = self.knowledge_encoder.initHidden()
                this_knowledge = knowledge[: query_idx]
                knowledge_output, knowledge_hidden = self.knowledge_encoder.forward(this_knowledge, knowledge_hidden,
                                                                                    data, query_idx)
                # attention for knowledge
                if self.use_knowledge_attention:
                    a = torch.mm(knowledge_output, torch.transpose(knowledge_hidden, 0, 1))
                    a = self.softmax1(a).view(-1, 1)
                    knowledge_memory = torch.mul(knowledge_output, a)
                    knowledge_output = knowledge_memory.sum(dim=0)
                else:
                    knowledge_output = knowledge_output[-1]
                batch_knowledge_output.append(knowledge_output)
            batch_knowledge_output = torch.stack(tuple(batch_knowledge_output), 0)
            knowledge_output_list.append(batch_knowledge_output)
        knowledge_output = torch.stack(tuple(knowledge_output_list), 0)

        # state encoding from interaction
        # interaction: batch_size * session_doc_num * data
        interaction_input_variable = interaction_variable
        interaction_input_variable = interaction_input_variable.cuda() if use_cuda else interaction_input_variable
        interaction_hidden = self.state_encoder.initHidden()

        # interaction_input_variable[:, :, i] has 4 parts: url, rank, vtype, click, each one is a one-hot vector
        interaction_output, interaction_hidden = self.state_encoder.forward(
            interaction_input_variable[:, :, 0], interaction_input_variable[:, :, 1],
            interaction_input_variable[:, :, 2],
            interaction_input_variable[:, :, 3], interaction_hidden, data)

        if self.use_state_attention:
            interaction_attention_output = []
            for batch_idx, batch_interaction in enumerate(interaction_output):
                batch_interaction_output = []
                for sess_pos_idx, interaction in enumerate(batch_interaction):
                    prev_hidden = interaction_output[batch_idx][: sess_pos_idx + 1]
                    interaction = interaction.view(1, -1)
                    a = torch.mm(interaction, torch.transpose(prev_hidden, 0, 1))
                    a = self.softmax2(a).view(-1, 1)

                    interaction_memory = torch.mul(prev_hidden, a)
                    this_interaction_output = interaction_memory.sum(dim=0)
                    batch_interaction_output.append(this_interaction_output)
                batch_interaction_output = torch.stack(tuple(batch_interaction_output), 0)
                interaction_attention_output.append(batch_interaction_output)
            interaction_output = torch.stack(tuple(interaction_attention_output), 0)

        # document encoding
        # document_input_variable has 3 parts: url, rank, vtype, each one is a one-hot vector
        document_input_variable = document_variable
        document_input_variable = document_input_variable.cuda() if use_cuda else document_input_variable
        document_output = self.document_encoder.forward(document_input_variable[:, :, 0],
                                                        document_input_variable[:, :, 1],
                                                        document_input_variable[:, :, 2],
                                                        document_input_variable[:, :, 3],
                                                        data)

        # concatenation and relevance estimator
        concat_output = torch.cat((knowledge_output, interaction_output, document_output), dim=2)
        relevance = self.relevance_estimator.forward(concat_output, self.batch_size)

        # examination prediction
        examination_input_variable = examination_context
        examination_input_variable = examination_input_variable.cuda() if use_cuda else examination_input_variable

        examination_list_output = []
        for batch_idx, batch_examination in enumerate(examination_input_variable):
            batch_examination_output = []
            query_num = batch_examination.size()[0] / 10
            for query_idx in range(query_num):
                this_query_context = batch_examination[query_idx * 10: (query_idx + 1) * 10]
                this_query_context = this_query_context.view(1, 10, -1)
                this_hidden = self.examination_predictor.initHidden()
                this_examination_output = self.examination_predictor.forward(this_query_context[:, :, 2],
                                                                             this_query_context[:, :, 3],
                                                                             this_query_context[:, :, 1], this_hidden)
                batch_examination_output.append(this_examination_output)
            batch_examination_output = torch.cat(tuple(batch_examination_output), 1)
            examination_list_output.append(batch_examination_output)
        examination_output = torch.cat(tuple(examination_list_output), 0)
        exam_prob = examination_output

        # combine the relevance and the examination according to the combination type
        clicks = self.get_clicks(relevance, exam_prob)
        return relevance, exam_prob, clicks
