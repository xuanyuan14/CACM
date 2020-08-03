# !/usr/bin/python
# coding: utf8

from xml.dom.minidom import parse
import xml.dom.minidom
import time
import pprint
import string
import sys
sys.path.append("..")
import argparse
import re
import os
import numpy as np
import torch
import torch.nn as nn
from utils import *
from math import log
import random

def xml_clean(args):
    # open xml file reader & writer
    xml_reader = open(os.path.join(args.input, args.dataset), 'r')
    xml_writer = open(os.path.join(args.input, 'clean-' + args.dataset), 'w')
    # print(xml_reader)
    # print(xml_writer)

    # remove useless lines
    read_line_count = 0
    removed_line_count = 0
    interaction_count = 0
    print('  - {}'.format('start reading from xml file...'))
    xml_lines = xml_reader.readlines()
    print('  - {}'.format('read {} lines'.format(len(xml_lines))))
    print('  - {}'.format('start removing useless lines...'))
    for xml_line in xml_lines:
        # print(xml_line, end='')
        read_line_count += 1
        if xml_line.find('<interaction num=') != -1:
            interaction_count += 1
        if xml_line_removable(xml_line):
            # A line that should be removed
            removed_line_count += 1
            if removed_line_count % 1000000 == 0:
                print('  - {}'.format('remove {} lines...'.format(removed_line_count)))
        else:
            xml_writer.write(xml_line)
    
    # It is guaranteed that there are 10 docs for each query
    assert read_line_count == len(xml_lines)
    assert removed_line_count == interaction_count + interaction_count * 10 * (1 + 1 + 2 + 6)
    print('  - {}'.format('read {} lines'.format(read_line_count)))
    print('  - {}'.format('totally {} iteractions'.format(interaction_count)))
    print('  - {}'.format('totally remove {} lines'.format(removed_line_count)))

def generate_dict_list(args):
    punc = '\\~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    session_sid = {}
    query_qid = {}
    url_uid = {}
    vtype_vid = {}
    uid_vid = {}

    print('  - {}'.format('start parsing xml file...'))
    DOMTree = xml.dom.minidom.parse(os.path.join(args.input, 'clean-' + args.dataset))
    tiangong2020 = DOMTree.documentElement
    sessions = tiangong2020.getElementsByTagName('session')
        
    # generate infos_per_session
    print('  - {}'.format('generating infos_per_session...'))
    infos_per_session = []
    junk_interation_num = 0
    for session in sessions:
        info_per_session = {}
        # get the session id
        session_number = int(session.getAttribute('num'))
        if not (session_number in session_sid):
            session_sid[session_number] = len(session_sid)
        info_per_session['session_number'] = session_number
        info_per_session['sid'] = session_sid[session_number]
        # print('session: {}'.format(session_number))
        
        # Get information within a query
        interactions = session.getElementsByTagName('interaction')
        interaction_infos = []
        for interaction in interactions:
            interaction_info = {}
            interaction_number = int(interaction.getAttribute('num'))
            query_id = interaction.getElementsByTagName('query_id')[0].childNodes[0].data
            if not (query_id in query_qid):
                query_qid[query_id] = len(query_qid)
            interaction_info['query'] = query_id
            interaction_info['qid'] = query_qid[query_id]
            interaction_info['session'] = info_per_session['session_number']
            interaction_info['sid'] = info_per_session['sid']
            # print('interaction: {}'.format(interaction_number))
            # print('query_id: {}'.format(query_id))

            # Get document infomation
            docs = interaction.getElementsByTagName('results')[0].getElementsByTagName('result')
            doc_infos = []
            if len(docs) == 0:
                print('  - {}'.format('WARNING: find a query with no docs: {}'.format(query_id)))
                junk_interation_num += 1
                continue
            elif len(docs) > 10:
                # more than 10 docs is not ok. May cause index out-of-range in embeddings
                print('  - {}'.format('WARNING: find a query with more than 10 docs: {}'.format(query_id)))
                junk_interation_num += 1
                continue
            elif len(docs) < 10:
                # less than 10 docs is ok. Never cause index out-of-range in embeddings
                print('  - {}'.format('WARNING: find a query with less than 10 docs: {}'.format(query_id)))
                junk_interation_num += 1
                continue
            for doc in docs:
                # WARNING: there might be junk data in TianGong-ST (e.g. rank > 10),  so we use manual doc_rank here
                doc_rank = int(doc.getAttribute('rank'))
                assert 1 <= doc_rank and doc_rank <= 10
                doc_id = doc.getElementsByTagName('docid')[0].childNodes[0].data
                vtype = doc.getElementsByTagName('vtype')[0].childNodes[0].data
                if not (doc_id in url_uid):
                    url_uid[doc_id] = len(url_uid)
                if not (vtype in vtype_vid):
                    vtype_vid[vtype] = len(vtype_vid)
                uid_vid[url_uid[doc_id]] = vtype_vid[vtype]
                doc_info = {}
                doc_info['rank'] = doc_rank
                doc_info['url'] = doc_id
                doc_info['uid'] = url_uid[doc_id]
                doc_info['vtype'] = vtype
                doc_info['vid'] = vtype_vid[vtype]
                doc_info['click'] = 0
                doc_infos.append(doc_info)
                # print('      doc ranks at {}: {}'.format(doc_rank, doc_id))

            # Get click information if there are clicked docs
            # Maybe there are no clicks in this query
            clicks = interaction.getElementsByTagName('clicked')
            if len(clicks) > 0:
                clicks = clicks[0].getElementsByTagName('click')
                for click in clicks:
                    clicked_doc_rank = int(click.getElementsByTagName('rank')[0].childNodes[0].data)
                    for item in doc_infos:
                        if item['rank'] == clicked_doc_rank:
                            item['click'] = 1
                            break
                    # print('      click doc ranked at {}'.format(clicked_doc_rank))
            else:
                pass
                # print('      click nothing')
            interaction_info['docs'] = doc_infos
            interaction_info['uids'] = [doc['uid'] for doc in doc_infos]
            interaction_info['vids'] = [doc['vid'] for doc in doc_infos]
            interaction_info['clicks'] = [doc['click'] for doc in doc_infos]
            interaction_infos.append(interaction_info)
        info_per_session['interactions'] = interaction_infos
        infos_per_session.append(info_per_session)
    print('  - {}'.format('abandon {} junk interactions'.format(junk_interation_num)))

    # generate infos_per_query
    print('  - {}'.format('generating infos_per_query...'))
    infos_per_query = []
    for info_per_session in infos_per_session:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            infos_per_query.append(interaction_info)

    # save and check infos_per_session
    print('  - {}'.format('save and check infos_per_session...'))
    print('    - {}'.format('length of infos_per_session: {}'.format(len(infos_per_session))))
    # pprint.pprint(infos_per_session)
    # print('length of infos_per_session: {}'.format(len(infos_per_session)))
    save_list(args.output, 'infos_per_session.list', infos_per_session)
    list1 = load_list(args.output, 'infos_per_session.list')
    assert len(infos_per_session) == len(list1)
    for idx, item in enumerate(infos_per_session):
        assert item == list1[idx]
    
    # save and check infos_per_query
    print('  - {}'.format('save and check infos_per_query...'))
    print('    - {}'.format('length of infos_per_query: {}'.format(len(infos_per_query))))
    # pprint.pprint(infos_per_query)
    # print('length of infos_per_query: {}'.format(len(infos_per_query)))
    save_list(args.output, 'infos_per_query.list', infos_per_query)
    list2 = load_list(args.output, 'infos_per_query.list')
    assert len(infos_per_query) == len(list2)
    for idx, item in enumerate(infos_per_query):
        assert item == list2[idx]
    
    # save and check dictionaries
    print('  - {}'.format('save and check session_sid, query_qid, url_uid...'))
    print('    - {}'.format('unique session number: {}'.format(len(session_sid))))
    print('    - {}'.format('unique query number: {}'.format(len(query_qid))))
    print('    - {}'.format('unique doc number: {}'.format(len(url_uid))))
    print('    - {}'.format('unique vtype number: {}'.format(len(vtype_vid))))
    save_dict(args.output, 'session_sid.dict', session_sid)
    save_dict(args.output, 'query_qid.dict', query_qid)
    save_dict(args.output, 'url_uid.dict', url_uid)
    save_dict(args.output, 'vtype_vid.dict', vtype_vid)
    save_dict(args.output, 'uid_vid.dict', uid_vid)

    dict1 = load_dict(args.output, 'session_sid.dict')
    dict2 = load_dict(args.output, 'query_qid.dict')
    dict3 = load_dict(args.output, 'url_uid.dict')
    dict4 = load_dict(args.output, 'vtype_vid.dict')
    dict5 = load_dict(args.output, 'uid_vid.dict')

    assert len(session_sid) == len(dict1)
    assert len(query_qid) == len(dict2)
    assert len(url_uid) == len(dict3)
    assert len(vtype_vid) == len(dict4)
    assert len(uid_vid) == len(dict5)

    for key in dict1:
        assert dict1[key] == session_sid[key]
        assert key > 0
    for key in dict2:
        assert dict2[key] == query_qid[key]
        assert key[0] == 'q'
        assert key[1:] != ''
    for key in dict3:
        assert dict3[key] == url_uid[key]
        assert key[0] == 'd'
        assert key[1:] != ''
    for key in dict4:
        assert dict4[key] == vtype_vid[key]
        assert type(key) == type('')
        assert key[1:] != ''
    for key in dict5:
        assert dict5[key] == uid_vid[key]
        assert key >= 0

    print('  - {}'.format('Done'))

def generate_txt(args):
    # load session_sid & query_qid & url_uid
    print('  - {}'.format('loading session_sid & query_qid & url_uid...'))
    session_sid = load_dict(args.output, 'session_sid.dict')
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')
    vtype_vid = load_dict(args.output, 'vtype_vid.dict')

    # write train.txt & dev.txt & test.txt per query
    print('  - {}'.format('loading the infos_per_session...'))
    infos_per_session = load_list(args.output, 'infos_per_session.list')
    # Separate all sessions into train : dev : test
    session_num = len(infos_per_session)
    train_dev_split = 117431
    dev_test_split = 117431 + 13154
    train_session_num = 117431
    dev_session_num = 13154
    test_session_num = session_num - train_session_num - dev_session_num
    print('    - {}'.format('train sessions: {}'.format(train_session_num)))
    print('    - {}'.format('dev sessions: {}'.format(dev_session_num)))
    print('    - {}'.format('test sessions: {}'.format(test_session_num)))
    print('    - {}'.format('total sessions: {}'.format(session_num)))
    
    # generate train & dev & test sessions
    print('  - {}'.format('generating train & dev & test data per sessions...'))
    '''random.seed(time.time())
    for _ in range(10):
        random.shuffle(infos_per_session)'''
    train_sessions = infos_per_session[:train_dev_split]
    dev_sessions = infos_per_session[train_dev_split:dev_test_split]
    test_sessions = infos_per_session[dev_test_split:]
    assert train_session_num == len(train_sessions), 'train_session_num: {}, len(train_sessions): {}'.format(train_session_num, len(train_sessions))
    assert dev_session_num == len(dev_sessions), 'dev_session_num: {}, len(dev_sessions): {}'.format(dev_session_num, len(dev_sessions))
    assert test_session_num == len(test_sessions), 'test_session_num: {}, len(test_sessions): {}'.format(test_session_num, len(test_sessions))
    assert session_num == len(train_sessions) + len(dev_sessions) + len(test_sessions), 'session_num: {}, len(train_sessions) + len(dev_sessions) + len(test_sessions): {}'.format(session_num, len(train_sessions) + len(dev_sessions) + len(test_sessions))

    # generate train & dev & test queries
    print('  - {}'.format('generating train & dev & test data per queries...'))
    train_queries = []
    dev_queries = []
    test_queries = []
    for info_per_session in train_sessions:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            train_queries.append(interaction_info)
    for info_per_session in dev_sessions:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            dev_queries.append(interaction_info)
    for info_per_session in test_sessions:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            test_queries.append(interaction_info)
    print('    - {}'.format('train queries: {}'.format(len(train_queries))))
    print('    - {}'.format('dev queries: {}'.format(len(dev_queries))))
    print('    - {}'.format('test queries: {}'.format(len(test_queries))))
    print('    - {}'.format('total queries: {}'.format(len(train_queries) + len(dev_queries) + len(test_queries))))
    
    # Write queries back to txt files
    print('  - {}'.format('writing queries back to txt files...'))
    print('    - {}'.format('writing into {}/train_per_query.txt'.format(args.output)))
    generate_data_per_query(train_queries, np.arange(0, len(train_queries)), args.output, 'train_per_query.txt')
    print('    - {}'.format('writing into {}/dev_per_query.txt'.format(args.output)))
    generate_data_per_query(dev_queries, np.arange(0, len(dev_queries)), args.output, 'dev_per_query.txt')
    print('    - {}'.format('writing into {}/test_per_query.txt'.format(args.output)))
    generate_data_per_query(test_queries, np.arange(0, len(test_queries)), args.output, 'test_per_query.txt')
    
    # Write sessions back to txt files
    print('  - {}'.format('writing sessions back to txt files...'))
    print('    - {}'.format('writing into {}/train_per_session.txt'.format(args.output)))
    generate_data_per_session(train_sessions, np.arange(0, len(train_sessions)), args.output, 'train_per_session.txt')
    print('    - {}'.format('writing into {}/dev_per_session.txt'.format(args.output)))
    generate_data_per_session(dev_sessions, np.arange(0, len(dev_sessions)), args.output, 'dev_per_session.txt')
    print('    - {}'.format('writing into {}/test_per_session.txt'.format(args.output)))
    generate_data_per_session(test_sessions, np.arange(0, len(test_sessions)), args.output, 'test_per_session.txt')

    # open human_labels.txt
    print('===> {}'.format('processing human label txt...'))
    label_reader = open(os.path.join(args.input + '../human_label/', 'sogou_st_human_labels.txt'), 'r')
    label_writer = open(os.path.join(args.output, 'human_label.txt'), 'w')

    # start transferring human labels
    read_line_count = 0
    write_line_count = 0
    print('  - {}'.format('start reading from human_label.txt...'))
    lines = label_reader.readlines()
    print('  - {}'.format('read {} lines'.format(len(lines))))
    print('  - {}'.format('start transferring...'))
    for line in lines:
        # there is a mixture of separators: ' ' & '\t'
        line_entry = [str(i) for i in line.strip().split()]
        read_line_count += 1
        # print(line_entry)
        line_entry[1] = str(session_sid[int(line_entry[1])])
        line_entry[2] = str(query_qid[line_entry[2]])
        line_entry[3] = str(url_uid[line_entry[3]])
        line_entry.append('\n')
        write_line_count += 1
        label_writer.write('\t'.join(line_entry))
    label_reader.close()
    label_writer.close()
    assert read_line_count == len(lines)
    assert write_line_count % 10 == 0
    print('  - {}'.format('write {} lines'.format(write_line_count)))
    print('  - {}'.format('finish reading from human_label.txt...'))
    print('  - {}'.format('Done'))

def generate_graph(args):
    # load dicts and lists
    print('  - {}'.format('loading query_qid, url_uid...'))
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')
    infos_per_session = load_list(args.output, 'infos_per_session.list')
    query_size = len(query_qid)
    doc_size = len(url_uid)
    print('  - VARIABLE: query_size-{}, doc_size-{}, num_session-{}'.format(query_size, doc_size, len(infos_per_session)))

    # generate qid_nid & uid_nid
    print('  - {}'.format('generating qid_nid & uid_nid...'))
    node_size = 0
    qid_nid = {}
    uid_nid = {}
    for key in query_qid:
        qid_nid[query_qid[key]] = node_size
        node_size += 1
    for key in url_uid:
        uid_nid[url_uid[key]] = node_size
        node_size += 1
    assert query_size + doc_size == node_size

    # save and check qid_nid & uid_nid
    print('  - {}'.format('saving qid_nid & uid_nid...'))
    save_dict(args.output, 'qid_nid.dict', qid_nid)
    save_dict(args.output, 'uid_nid.dict', uid_nid)
    dict1 = load_dict(args.output, 'qid_nid.dict')
    dict2 = load_dict(args.output, 'uid_nid.dict')
    assert len(qid_nid) == len(dict1)
    assert len(uid_nid) == len(dict2)
    for key in qid_nid:
        assert dict1[key] == qid_nid[key]
    for key in uid_nid:
        assert dict2[key] == uid_nid[key]

    # generate graph paths
    print('  - {}'.format('generating q2q, u2u, q2u paths...'))
    graph_file = open(os.path.join(args.output, 'TianGong-ST.edgelist'), 'w')
    for info_per_session in infos_per_session:
        interaction_infos = info_per_session['interactions']
        # generate query-query path
        q2q_path_weight = 2.0
        for cur_interaction_idx in range(1, len(interaction_infos)):
            cur_qid = interaction_infos[cur_interaction_idx]['qid']
            prev_qid = interaction_infos[cur_interaction_idx - 1]['qid']
            graph_file.write('{} {} {}\n'.format(qid_nid[prev_qid], qid_nid[cur_qid], q2q_path_weight))
        # generate url-url path
        for interaction_info in interaction_infos:
            doc_infos = interaction_info['docs']
            for cur_doc_idx in range(1, len(doc_infos)):
                cur_uid = doc_infos[cur_doc_idx]['uid']
                prev_uid = doc_infos[cur_doc_idx - 1]['uid']
                u2u_path_weight = 1 / log(doc_infos[cur_doc_idx]['rank'], 2)
                graph_file.write('{} {} {}\n'.format(uid_nid[prev_uid], uid_nid[cur_uid], u2u_path_weight))
        # generate query-url path
        for interaction_info in interaction_infos:
            cur_qid = interaction_info['qid']
            doc_infos = interaction_info['docs']
            last_click_rank = 11 # in case there are no clicks in a query
            for doc_info in doc_infos:
                if doc_info['click'] == 1:
                    last_click_rank = doc_info['rank']
            for doc_info in doc_infos:
                cur_uid = doc_info['uid']
                if doc_info['click'] == 1:
                    q2u_path_weight = 1.0
                elif doc_info['rank'] < last_click_rank:
                    q2u_path_weight = -1.0
                else:
                    q2u_path_weight = 0.0
                graph_file.write('{} {} {}\n'.format(qid_nid[cur_qid], uid_nid[cur_uid], q2u_path_weight))
    print('  - {}'.format('Done'))

def generate_human_label_txt_for_CACM(args):
    print('  - {}'.format('loading infos_per_session.list'))
    infos_per_session = load_list(args.output, 'infos_per_session.list')
    
    print('  - {}'.format('parse human_label.txt and generate relevance queries'))
    label_reader = open(os.path.join(args.output, 'human_label.txt'), "r")
    relevance_queries = []
    query_count = dict()
    previous_id = -1
    cnt = 0
    for line in label_reader:
        entry_array = line.strip().split()
        id = int(entry_array[0])
        task = int(entry_array[1])
        query = int(entry_array[2])
        result = int(entry_array[3])
        relevance = int(entry_array[4])
        
        # count query-doc pairs
        if not query in query_count:
            query_count[query] = dict()
            query_count[query][result] = 1
        elif not result in query_count[query]:
            query_count[query][result] = 1
        else:
            query_count[query][result] += 1

        # The first line of a sample query
        if id != previous_id:
            info_per_query = dict()
            info_per_query['id'] = id
            info_per_query['sid'] = task
            info_per_query['qid'] = query
            info_per_query['uids'] = [result]
            info_per_query['relevances'] = [relevance]
            relevance_queries.append(info_per_query)
            cnt += 1
            previous_id = id
        
        # The rest lines of a query
        else:
            relevance_queries[-1]['uids'].append(result)
            relevance_queries[-1]['relevances'].append(relevance)
            cnt += 1
    tmp = 0
    for key in query_count:
        for x in query_count[key]:
            tmp += query_count[key][x]
    assert tmp == 20000
    assert cnt == 20000
    print('  - num of queries in human_label.txt: {}'.format(len(relevance_queries)))
    
    print('  - {}'.format('saving the relevance queries'))
    save_list(args.output, 'relevance_queries.list', relevance_queries)
    list1 = load_list(args.output, 'relevance_queries.list')
    assert len(relevance_queries) == len(list1)
    for idx, item in enumerate(relevance_queries):
        assert item == list1[idx]
    
    # NOTE: need to resort the doc within a query to get the click infos
    generate_data_per_session_for_human_label(relevance_queries, infos_per_session, np.arange(0, len(relevance_queries)), args.output, 'human_label_for_CACM.txt')

def main():
    parser = argparse.ArgumentParser('TianGong-ST')
    parser.add_argument('--dataset', default='sogousessiontrack2020.xml',
                        help='dataset name')
    parser.add_argument('--input', default='../dataset/TianGong-ST/data/',
                        help='input path')
    parser.add_argument('--output', default='./data/CACM',
                        help='output path')
    parser.add_argument('--xml_clean', action='store_true',
                        help='remove useless lines in xml files, to reduce the size of xml file')
    parser.add_argument('--dict_list', action='store_true',
                        help='generate dicts and lists for info_per_session/info_per_query')
    parser.add_argument('--txt', action='store_true',
                        help='generate NCM data txt')
    parser.add_argument('--node2vec', action='store_true',
                        help='generate graph data for node2vec')
    parser.add_argument('--human_label_txt_for_CACM', action='store_true',
                        help='generate human_label_txt_for_CACM.txt')
    parser.add_argument('--trainset_ratio', default=0.8,
                        help='ratio of the train session/query according to the total number of sessions/queries')
    parser.add_argument('--devset_ratio', default=0.1,
                        help='ratio of the dev session/query according to the total number of sessions/queries')
    args = parser.parse_args()

    if args.xml_clean:
        # remove useless lines in xml files, to reduce the size of xml file
        print('===> {}'.format('cleaning xml file...'))
        xml_clean(args)
    if args.dict_list:
        # generate info_per_session & info_per_query
        print('===> {}'.format('generating dicts and lists...'))
        generate_dict_list(args)
    if args.txt:
        # load lists saved by generate_dict_list() and generates train.txt & dev.txt & test.txt
        print('===> {}'.format('generating train & dev & test data txt...'))
        generate_txt(args)
    if args.node2vec:
        # generate graph data for node2vec
        print('===> {}'.format('generating graph data for node2vec...'))
        generate_graph(args)
    if args.human_label_txt_for_CACM:
        # generate human label txt for CACM
        print('===> {}'.format('generating human label txt for CACM...'))
        generate_human_label_txt_for_CACM(args)
    print('===> {}'.format('Done.'))
    
if __name__ == '__main__':
    main()