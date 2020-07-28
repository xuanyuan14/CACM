'''
@Author: your name
@Date: 2020-07-09 11:47:56
@LastEditTime: 2020-07-25 19:35:50
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model/utils.py
'''
import os
import pprint

def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def save_dict(file_path, file_name, dict):
    check_path(file_path)
    data_path = os.path.join(file_path, file_name)
    file = open(data_path, 'w')
    file.write(str(dict))
    file.close()

def load_dict(file_path, file_name):
    data_path = os.path.join(file_path, file_name)
    assert os.path.isfile(data_path), '{} file does not exist.'.format(data_path)
    file = open(data_path, 'r')
    return eval(file.read())
    
def save_list(file_path, file_name, list_data):
    check_path(file_path)
    data_path = os.path.join(file_path, file_name)
    file = open(data_path, 'w')
    file.write(str(list_data))
    file.close()

def load_list(file_path, file_name):
    data_path = os.path.join(file_path, file_name)
    assert os.path.isfile(data_path), '{} file does not exist.'.format(data_path)
    file = open(data_path, 'r')
    return eval(file.read())

def generate_data_per_query(infos_per_query, indices, file_path, file_name):
    check_path(file_path)
    data_path = os.path.join(file_path, file_name)
    file = open(data_path, 'w')
    for key in indices:
        interaction_info = infos_per_query[key]
        sid = interaction_info['sid']
        qid = interaction_info['qid']
        uids = interaction_info['uids']
        vids = interaction_info['vids']
        clicks = interaction_info['clicks']
        file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(sid, qid, 0, 0, str(uids), str(vids), str(clicks)))
        
        file.write('\n')
    file.close()

def generate_data_per_session(infos_per_session, indices, file_path, file_name):
    check_path(file_path)
    data_path = os.path.join(file_path, file_name)
    file = open(data_path, 'w')
    for key in indices:
        query_sequence_for_print = []
        prev_document_info_for_print = []
        info_per_session = infos_per_session[key]
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            qid = interaction_info['qid']
            uids = interaction_info['uids']
            vids = interaction_info['vids']
            clicks = interaction_info['clicks']
            query_sequence_for_print.append(qid)
            for idx, uid in enumerate(uids):
                vid = vids[idx]
                click = clicks[idx]
                rank = idx + 1
                document_info_for_print = [uid, rank, vid] 
                file.write('{}\t{}\t{}\t{}\n'.format(str(query_sequence_for_print), 
                                                     str(prev_document_info_for_print), 
                                                     str(document_info_for_print),
                                                     click))
                prev_document_info_for_print = [uid, rank, vid, click]
        file.write('\n')
    file.close()

def xml_line_removable(xml_line):
    if xml_line.find('<query>') != -1 and xml_line.find('</query>') != -1:
        return 1
    elif xml_line.find('<url>') != -1 and xml_line.find('</url>') != -1:
        return 1
    elif xml_line.find('<title>') != -1 and xml_line.find('</title>') != -1:
        return 1
    elif xml_line.find('<relevance>') != -1 or xml_line.find('</relevance>') != -1:
        return 1
    elif xml_line.find('<TACM>') != -1 and xml_line.find('</TACM>') != -1:
        return 1
    elif xml_line.find('<PSCM>') != -1 and xml_line.find('</PSCM>') != -1:
        return 1
    elif xml_line.find('<THCM>') != -1 and xml_line.find('</THCM>') != -1:
        return 1
    elif xml_line.find('<UBM>') != -1 and xml_line.find('</UBM>') != -1:
        return 1
    elif xml_line.find('<DBN>') != -1 and xml_line.find('</DBN>') != -1:
        return 1
    elif xml_line.find('<POM>') != -1 and xml_line.find('</POM>') != -1:
        return 1
    return 0
    