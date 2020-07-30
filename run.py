#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ref: A Context-Aware Click Model for Web Search
@author: Jia Chen, Jiaxin Mao, Yiqun Liu, Min Zhang, Shaoping Ma
@desc: Configurations and startups
'''
import os
import argparse
import logging
import time
from dataset import Dataset
from model import Model
from utils import *

def parse_args():
    parser = argparse.ArgumentParser('CACM')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--test', action='store_true',
                        help='test on test set')
    parser.add_argument('--gpu', type=str, default='',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adadelta',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=1e-3,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.2,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=1,
                                help='train batch size')
    train_settings.add_argument('--num_steps', type=int, default=20000,
                                help='number of training steps')
    train_settings.add_argument('--num_train_files', type=int, default=1,
                                help='number of training files')
    train_settings.add_argument('--num_dev_files', type=int, default=1,
                                help='number of dev files')
    train_settings.add_argument('--num_test_files', type=int, default=1,
                                help='number of test files')
    train_settings.add_argument('--num_label_files', type=int, default=1,
                                help='number of label files')
    train_settings.add_argument('--reg_relevance', type=float, default=1.0,
                                help='regularization for relevance training')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='CACM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=128,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=256,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_d_num', type=int, default=10,
                                help='max number of docs in a session')
    model_settings.add_argument('--max_sess_length', type=int, default=10,
                                help='max session length')
    model_settings.add_argument('--use_knowledge', type=bool, default=False,
                                help='whether use knowledge embedding')
    model_settings.add_argument('--use_knowledge_attention', type=bool, default=False,
                                help='whether use knowledge attention')
    model_settings.add_argument('--use_state_attention', type=bool, default=False,
                                help='whether use state attention')
    model_settings.add_argument('--combine', default='mul',
                                help='type of combining the relevance and the examination to predict the click')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_dirs', nargs='+',
                                default=['data/CACM/train_per_session.txt'],
                                help='list of dirs that contain the preprocessed train data')
    path_settings.add_argument('--dev_dirs', nargs='+',
                                default=['data/CACM/dev_per_session.txt'],
                                help='list of dirs that contain the preprocessed dev data')
    path_settings.add_argument('--test_dirs', nargs='+',
                                default=['data/CACM/test_per_session.txt'],
                                help='list of dirs that contain the preprocessed test data')
    path_settings.add_argument('--label_dirs', nargs='+',
                                default=['data/CACM/human_label_for_CACM.txt'],
                                help='list of dirs that contain the preprocessed label data')
    path_settings.add_argument('--knowledge_type', default='simple',
                                help='type of knowledge embedding')
    path_settings.add_argument('--data_dir', default='outputs/CACM/',
                                help='the main dir')
    path_settings.add_argument('--model_dir', default='outputs/CACM/models/',
                                help='the dir to store models')
    path_settings.add_argument('--result_dir', default='outputs/CACM/results/',
                                help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='outputs/CACM/summary/',
                                help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_dir', default='outputs/CACM/log/',
                                help='path of the log file. If not set, logs are printed to console')

    path_settings.add_argument('--eval_freq', type=int, default=2000,
                                help='the frequency of evaluating on the dev set when training')
    path_settings.add_argument('--check_point', type=int, default=2000,
                                help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=3,
                                help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
    path_settings.add_argument('--lr_decay', type=float, default=0.5,
                                help='lr decay')
    path_settings.add_argument('--load_model', type=int, default=-1,
                                help='load model global step')
    path_settings.add_argument('--data_parallel', type=bool, default=False,
                                help='data_parallel')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                                help='gpu_num')

    return parser.parse_args()

def test(args):
    logger = logging.getLogger("CACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs + args.label_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs, label_dirs=args.label_dirs)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_nid), len(dataset.uid_nid), len(dataset.vtype_vid))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Start computing LL & PPL for click prediction...')
    test_batches = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    test_loss, test_LL, test_perplexity, test_perplexity_at_rank = model.evaluate(test_batches, dataset, result_dir=args.result_dir,
                                                                                    result_prefix='test.predicted.{}.{}'.format(args.algo, model.global_step))
    logger.info('Loss: {}'.format(test_loss))
    logger.info('log likelihood: {}'.format(test_LL))
    logger.info('perplexity: {}'.format(test_perplexity))
    logger.info('Start computing NDCG@k for ranking performance')
    label_batches = dataset.gen_mini_batches('label', args.batch_size, shuffle=False)
    trunc_levels = [1, 3, 5, 10]
    ndcgs_version1, ndcgs_version2 = model.ndcg(label_batches, dataset, result_dir=args.result_dir,
                                                result_prefix='test.rank.{}.{}'.format(args.algo, model.global_step))
    for trunc_level in trunc_levels:
        ndcg_version1, ndcg_version2 = ndcgs_version1[trunc_level], ndcgs_version2[trunc_level]
        logger.info("NDCG@{}: {}, {}".format(trunc_level, ndcg_version1, ndcg_version2))
    logger.info('Done with model testing!')

def train(args):
    logger = logging.getLogger("CACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs + args.label_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs, label_dirs=args.label_dirs)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_nid), len(dataset.uid_nid), len(dataset.vtype_vid))
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_model > -1:
        logger.info('Restoring the model...')
        model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    model.train(dataset)
    logger.info('Done with model training!')

def run():
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.hidden_size % 2 == 0
    logger = logging.getLogger("CACM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    check_path(args.model_dir)
    check_path(args.result_dir)
    check_path(args.summary_dir)
    if args.log_dir:
        check_path(args.log_dir)
        file_handler = logging.FileHandler(os.path.join(args.log_dir, time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())) + '.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    logger.info('Checking the directories...')
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.rank:
        rank(args)
    logger.info('run done.')


if __name__ == '__main__':
    run()
