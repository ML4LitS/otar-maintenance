import sys
import glob
from bs4 import BeautifulSoup
import lxml
from collections import defaultdict
from tqdm import tqdm
import requests
import random
import sys
import pathlib
import csv
import pandas as pd
import json
import argparse
import datetime

# import multiprocessing
from fuzzywuzzy import fuzz
from statistics import mean
import numpy as np
import itertools
import re
import io

# BioBERT NER models
import torch
from torch.utils.data import DataLoader
import pickle
from biobert.model.bert_crf_model import BertCRF
from biobert.data_loader.epmc_loader import NERDatasetBatch
from biobert.utils.utils import my_collate

from collections import namedtuple


# Relations and associations model
# import en_ner_europepmc_md
# import en_relationv01

import unicodedata
import datetime

from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

"""
This scripts use machine learning model as a filter to remove potential false positives.
It can run on EBI clusters. For each annotated gzip file, set up a machine learning model on each computer node
and do ML filtering. 
"""
Entity = namedtuple('Entity', ['span', 'tag', 'text', 'pre', 'post'])
Entity_Label = namedtuple('Label', ['index', 'pos', 'tag', 'span'])
missing_list = ['covid-19', 'coronavirus disease 2019', '2019-ncov', 'covid 19']


# Define all functions here

class MLModel:
    def __init__(self):
        self.bertCrf_model = load_model()
        self.bertCrf_model.load_state_dict(torch.load(MODEL_PATH + 'bert_crf_model.states', map_location=device))
        self.bertCrf_model.bert_model.bert_model.to(device)

    def post(self, sentences):
        BATCH_SIZE = 8
        text = sentences
        # print(text)
        with torch.no_grad():
            processor, tokens, spans = load_data_processor(text)
            dataLoader = DataLoader(dataset=processor, batch_size=BATCH_SIZE, collate_fn=my_collate, num_workers=2)

            idx2label = params['idx2label']
            self.bertCrf_model.eval()
            entities = []
            for i_batch, sample_batched in enumerate(dataLoader):
                inputs = sample_batched['input']

                bert_inputs, bert_attention_mask, bert_token_mask, wordpiece_alignment, split_alignments, lengths, token_mask \
                    = processor.tokens_totensor(inputs)

                _, preds = self.bertCrf_model.predict(input_ids=bert_inputs.to(device),
                                                      bert_attention_mask=bert_attention_mask.to(device),
                                                      bert_token_mask=bert_token_mask,
                                                      alignment=wordpiece_alignment,
                                                      splits=(split_alignments, lengths),
                                                      token_mask=token_mask)
                if idx2label:
                    for i, (path, score) in enumerate(preds):
                        labels = [idx2label[p] for p in path]
                        offset_index = i_batch * BATCH_SIZE + i
                        entities.append([[e.span[0], e.span[1], e.tag, e.text]
                                         for e in extract_entity(labels, spans[offset_index], text[offset_index])])
        return {'annotations': entities}


def load_data_processor(inputs):
    token_spans = []
    tokens = []
    for line in inputs:
        token_spans.append(list(tokenizer.span_tokenize(line)))
        tokens.append([line[start: end] for start, end in token_spans[-1]])

    processor = NERDatasetBatch.from_params(params=params, inputs=tokens)
    return processor, tokens, token_spans


def load_model():
    allowed_transitions = None
    model = BertCRF(num_tags=params['num_tags'],
                    model_name=params['model_name'],
                    stride=params['stride'],
                    include_start_end_transitions=True,
                    constraints=allowed_transitions)
    return model


def extract_entity(preds, spans, text, length=20):
    """
    extract entity from label sequence
    :param preds: a list of labels in a sentence
    :type preds: List[str
    :param spans:
    :type spans:
    :return: A list of entity object
    :rtype: List[Entity]
    """
    entities = []
    tmp = []

    for i, token in enumerate(preds):
        if token == 'O':
            pos, tag = 'O', 'O'
            label = None
        else:
            pos, tag = token.split('-')
            label = Entity_Label(index=i, pos=pos, tag=tag, span=spans[i])

        if pos in {'B', 'O'} and tmp:
            start_span = tmp[0].span[0]
            end_span = tmp[-1].span[1]
            entities.append(Entity(span=(start_span, end_span),
                                   tag=tmp[0].tag,
                                   text=text[start_span:end_span],
                                   pre=text[max(0, start_span - length):start_span],
                                   post=text[end_span: end_span + length]))
            tmp[:] = []
        if pos == 'B' or pos == 'I':
            tmp.append(label)

    if tmp:
        start_span = tmp[0].span[0]
        end_span = tmp[-1].span[-1]
        entities.append(
            Entity(span=(start_span, end_span),
                   tag=tmp[0].tag,
                   text=text[start_span:end_span],
                   pre=text[max(0, start_span - length):start_span],
                   post=text[end_span:end_span + length])
        )
    return entities


# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def clean_Nones(ner_tags_):
    ner_tags = []
    # had to do this as the position of entity tag and entity are exchanged in CD
    for each_ner_tag in ner_tags_:
        if 'CD' == each_ner_tag[2]:
            ner_tags.append([each_ner_tag[0], each_ner_tag[1], each_ner_tag[3], each_ner_tag[2]])
        else:
            ner_tags.append(each_ner_tag)

    ner_tags = sorted(ner_tags, key=lambda x: len(x[3]), reverse=True)
    if len(ner_tags) == 1 and 'None' in ner_tags:
        return ner_tags
    elif len(ner_tags) > 1 and 'None' in ner_tags:
        ner_tags.remove('None')
        return ner_tags
    else:
        return ner_tags



# This function will compare ml tags and ztags. The agreed tags are then returned back
def compare_ml_annotations_with_dictionary_tagged(ml_tags_, z_tags_, missing_list_):
    agreed_z_tags = set()
#     print(z_tags_, ml_tags_)
    for each_z_tag in z_tags_:
        for each_ml_annotation in ml_tags_:
            if each_z_tag.lower() in missing_list_:
                agreed_z_tags.add(each_z_tag)
            else:
                score = fuzz.partial_ratio(each_ml_annotation, each_z_tag) #token_set_ratio
                if score > 80:
                    agreed_z_tags.add(each_z_tag)
    return agreed_z_tags


fulltext_scores = {
    'title': 10,
    'intro': 1,
    'result': 5,
    'discus': 2,
    'conclus': 2,
    'case': 1,
    'fig': 5,
    'table': 5,
    'appendix': 1,
    'other': 1
}


# fulltext_scores
def assign_scores_to_sections(fulltext_scores_, section_tagged_):
    scores = []
    for key, val in fulltext_scores_.items():
        if key in section_tagged_.lower():
            return val

    return fulltext_scores_['other']



# read xmls files
def getfileblocks(file_path, document_flag):
    sub_file_blocks = []
    if document_flag == 'f':
        start_str1 = '<articles><article '
        start_str2 = '<article '
        try:
            with io.open(file_path, 'r', encoding="utf8") as fh:  #, encoding='utf8'
                for line in fh:
                    if line.startswith(start_str2) or line.startswith(start_str1):
                        sub_file_blocks.append(line.replace('^<articles>', ''))
                    else:
                        sub_file_blocks[-1] += line.strip().replace('</articles>$', '')
        except:
            with io.open(file_path, 'r', encoding="ISO-8859-1") as fh:  #, encoding='utf8'
                for line in fh:
                    if line.startswith(start_str2) or line.startswith(start_str1):
                        sub_file_blocks.append(line.replace('^<articles>', ''))
                    else:
                        sub_file_blocks[-1] += line.strip().replace('</articles>$', '')
    elif document_flag == 'a':
        start_str1 = '<article>'
        start_str2 = '<articles>'
        with io.open(file_path, 'rt', encoding='utf8') as fh:
            for line in fh:
                if line.startswith(start_str1):
                    sub_file_blocks.append(line)
                elif line.startswith(start_str2):
                    continue
                else:
                    sub_file_blocks[-1] += line
    else:
        print('ERROR: unknown document type :' + document_flag)
        sys.error(1)

    return sub_file_blocks


# this function will generate the tag spans given the missing spans of entities
def get_new_missing_tags(each_sentence, missing_list_, tag_type):
    new_entities = []
    for missing_string in missing_list_:
        for i in re.finditer(missing_string, each_sentence):
            indexlocation= i.span()
    #         print(indexlocation)
            startindex= i.start()
            endindex= i.end()
            entity = each_sentence[indexlocation[0]:indexlocation[1]]
            new_entities.append([startindex,endindex, tag_type, entity])
    return new_entities


# this will get matches
def get_sentences_matches_tags(sentences_tags, abs_full):
    matches = defaultdict(list)
    for each_sentence, ml_tags in sentences_tags.items():
        for each_ml_tag in ml_tags:
            if each_ml_tag[2] != 'OG':
                mini_dict = {}
                mini_dict['label'] = each_ml_tag[3]
                mini_dict['type'] = each_ml_tag[2]
                mini_dict['startInSentence'] = each_ml_tag[0]
                mini_dict['endInSentence'] = each_ml_tag[1]
                if each_sentence in abs_full:
                    start_index = abs_full.find(each_sentence)
                    mini_dict['sectionStart'] = start_index
                    mini_dict['sectionEnd'] = start_index + len(each_sentence)
                matches[each_sentence].append(mini_dict)
    return matches


# map annotations in sets of pairs
def get_mapped_list_from_annotations(annotation_list):
    mapped_list = list(itertools.combinations(annotation_list, 2))

    unique_maplist = []
    for each_list in mapped_list:
        if each_list[0][2] != each_list[1][2] and each_list[1][2] != 'OG' and each_list[0][2] != 'OG':
            unique_maplist.append((each_list[0], each_list[1]))

    return unique_maplist


# get only those sentences with relevant pairs
def get_sentences_offset_per_cooccurance(sentences_tags):
    dict_gp_ds = defaultdict(list)
    dict_gp_cd = defaultdict(list)
    dict_ds_cd = defaultdict(list)

    for sentence, tags in sentences_tags.items():
        if len(tags) > 1:  # only if more than 1 tag is available
            check_tags = np.array(tags)
            if 'GP' in check_tags and 'DS' in check_tags:
                dict_gp_ds[sentence] = get_mapped_list_from_annotations(tags)
            if 'GP' in check_tags and 'CD' in check_tags:
                dict_gp_cd[sentence] = get_mapped_list_from_annotations(tags)
            if 'DS' in check_tags and 'CD' in check_tags:
                dict_ds_cd[sentence] = get_mapped_list_from_annotations(tags)

    return dict_gp_ds, dict_gp_cd, dict_ds_cd


# if not in the right position if the pair and swap them such that always GP is followed by either CD or DS and DS is followed by CD
def swap_positions(cooccurance_list, pos1, pos2):
    cooccurance_list[pos1], cooccurance_list[pos2] = cooccurance_list[pos2], cooccurance_list[pos1]
    return cooccurance_list



# this is for getting relationship text
def get_relations(gp_ds_text_sentence):
    docs = relation_model2(gp_ds_text_sentence)
    rel_list =[]
    for ent in docs.ents:
        if ent.label_!='GP' and ent.label_!='DS':
            rel_dict = {}
            rel_dict['startr'] = ent.start_char
            rel_dict['endr'] = ent.end_char
            rel_dict['labelr'] = ent.text
            rel_dict['typer'] = ent.label_
            rel_list.append(rel_dict)
    return rel_list

# roundoff the association model scores
def roundoff(dict_y):
    for k, v in dict_y.items():
        v = round(v,2)
        dict_y[k] = v
    return dict_y


# get the occurances
def get_cooccurance_evidence(average_evidence_scores, dict_tags, tag_type_1, tag_type_2):
    co_occurance_sentences = defaultdict(list)
    #     mined_sentences = []
    for each_sent_map, mappedtags in dict_tags.items():
        # always see that GP-DS, GP-CD and CD-DS is followed
        if tag_type_1 not in mappedtags[0][0][2]:
            mappedtags[0] = swap_positions(list(mappedtags[0]), 0, 1)
        else:
            mappedtags[0] = list(mappedtags[0])
        for eachtag in mappedtags:
            if tag_type_1 == eachtag[0][2] and tag_type_2 == eachtag[1][2]:
                mini_dict = {}
                mini_dict['start1'] = eachtag[0][0]
                mini_dict['end1'] = eachtag[0][1]
                mini_dict['label1'] = eachtag[0][3]
                mini_dict['start2'] = eachtag[1][0]
                mini_dict['end2'] = eachtag[1][1]
                mini_dict['label2'] = eachtag[1][3]
                mini_dict['type'] = tag_type_1 + '-' + tag_type_2

                if average_evidence_scores[each_sent_map]:
                    mini_dict['sentEvidenceScore'] = average_evidence_scores[each_sent_map]
                else:
                    mini_dict['sentEvidenceScore'] = 1
                if tag_type_1 == 'GP' and tag_type_2 == 'DS':
                    # get associations scores
                    mini_dict['association'] = 0 #roundoff(relation_model1(each_sent_map).cats)
                    # get relations
                    rels = None #get_relations(each_sent_map)
                    if rels:
                        mini_dict['relation'] = rels
                co_occurance_sentences[each_sent_map].append(mini_dict)
    return co_occurance_sentences


def get_ml_tags(all_sentences, missing_list_):
    ML_annotations = ml_model.post(all_sentences)
    # Biobert is missing COVIS-19, need to retrain the model later. For now I tag it as DS
    final_annotations = []
    for each_annotation in ML_annotations['annotations']:
        if each_annotation:  # Biobert is tagging COVID-19 as GP need to retrain the model later. For now I tag it as DS
            if each_annotation[0][2] == 'GP' and each_annotation[0][3].lower() in missing_list_:
                each_annotation[0][2] = 'DS'
                final_annotations.append(each_annotation)
            elif each_annotation[0][2] == 'CD' and each_annotation[0][3].lower() == 'and':
                final_annotations.append(each_annotation)
            else:
                final_annotations.append(each_annotation)
        else:
            final_annotations.append(each_annotation)

    return final_annotations


def get_only_ml_tagged_sentences(sentences, ml_annots, missing_list_):
    gp_set = set()
    ds_set = set()
    cd_set = set()
    og_set = set()
    ml_tagged_sentences = {}
    count = 0

    for each_sentence in sentences:
        new_entities = get_new_missing_tags(each_sentence, missing_list_, tag_type='DS')
        all_tags = new_entities + ml_annots[count]

        if all_tags:
            ml_tagged_sentences[each_sentence] = all_tags
            for each_ml_tag in all_tags:
                if each_ml_tag[2] == 'GP':
                    gp_set.add(each_ml_tag[3])
                elif each_ml_tag[2] == 'DS':
                    ds_set.add(each_ml_tag[3])
                if each_ml_tag[2] == 'CD':
                    cd_set.add(each_ml_tag[3])
                if each_ml_tag[2] == 'OG':
                    og_set.add(each_ml_tag[3])
        count = count + 1
    return ml_tagged_sentences, gp_set, cd_set, og_set


def get_only_ztag_sentences(sentences, uniprot_fp_removed_set, z_efo_set, cd_set):
    new_cd_set = set()
    ztag_sentences = {}

    for each_cd_tag in cd_set:
        if 'and' != each_cd_tag:
            new_cd_set.add(each_cd_tag.replace(')', '').replace('(', '').strip())

    for each_sentence in sentences:
        uniport_entities = get_new_missing_tags(each_sentence, uniprot_fp_removed_set, tag_type='GP')
        efo_entities = get_new_missing_tags(each_sentence, z_efo_set, tag_type='DS')
        try:
            cd_entities = get_new_missing_tags(each_sentence, new_cd_set, tag_type='CD')
        except:
            cd_entities = []

        all_tags = uniport_entities + efo_entities + cd_entities

        if all_tags:
            ztag_sentences[each_sentence] = all_tags

    return ztag_sentences


def extract_sentence_level_details(soup, document_flag):
    plain_sentences_ = []
    section_tags_ = defaultdict(set)
    evidence_scores_ = defaultdict(list)
    average_evidence_scores__ = defaultdict(list)
    uniprot_set_ = set()
    efo_set_ = set()
    line_count = 0
    #print('In extract_sentence_level_details')
    # get all the sentences, UPDATE : except for the refernce sentences, change code
    all_sentences = soup.find_all('sent')
    '''
    # get uniprot tags
    try:
        uniprot_ztags = soup.find_all('z:uniprot')
        for each_tag in uniprot_ztags:
            uniprot_set_.add(each_tag.text)
    except:
        print('no uniprot_ztags found ')
    # get efo tags
    try:
        efo_ztags = soup.find_all('z:efo')
        for each_tag in efo_ztags:
            efo_set_.add(each_tag.text)
    except:
        print('no efo_ztags found ')
    '''
    # get abstract details if found
    try:
        abs_full = soup.find('abstract').text
        abs_sentences = soup.find('abstract').find_all('plain')
        total_abstract_length = len(abs_sentences)
    except:
        abs_full = ''
        abs_sentences = ''
        total_abstract_length =0

    # get section tags, evidence_scores_ and plain sentences
    for each_sentence in all_sentences:
        extracted_sentence = each_sentence.plain
        if extracted_sentence:
            #print(extracted_sentence)
            clean_text = unicodedata.normalize("NFKD", extracted_sentence.text).strip()

            try:
                if document_flag == 'f':
                    title_tag = extracted_sentence.findParent('article-title')
                elif document_flag == 'a':
                    title_tag = extracted_sentence.findParent('title')
            except:
                title_tag = ''

            try:
                if title_tag:
                    section_tags_[clean_text].add('title')
                    evidence_scores_[clean_text].append(10)
                    plain_sentences_.append(clean_text)
                else:
                    try:
                        if document_flag == 'f':
                            section_tagged = 'OTHER'
                            #print("In find Parent, i.e. SecTag")
                            #print(extracted_sentence.parent.name)
                            sec_parent = extracted_sentence.findParent('sectag')
                            if sec_parent:
                                #print('SecParent found')
                                section_tagged = sec_parent.get('type')
                            #print("section_tagged: " + section_tagged)
                        elif document_flag == 'a':
                            if extracted_sentence in abs_sentences:
                                section_tagged = 'Abstract'
                    except:
                        section_tagged = 'OTHER'
                        print('SecTag not found: ' + extracted_sentence)
                        
                    #print('section tag:' + section_tagged)
                    if section_tagged and section_tagged!='REF':
                        section_tags_[clean_text].add(section_tagged)
                        #print('in section tag and its not REF :' + section_tagged)
                        # evidence scores
                        if 'abstract' in section_tagged.lower():
                            #print(line_count)
                            line_count = line_count + 1
                            if line_count == 1 or line_count == 2:
                                evidence_scores_[clean_text].append(2)
                            elif line_count == total_abstract_length:
                                evidence_scores_[clean_text].append(5)
                            else:
                                evidence_scores_[clean_text].append(3)
                        else:
                            if document_flag == 'f':
                                evi_scor = assign_scores_to_sections(fulltext_scores, section_tagged)
                            elif document_flag == 'a':
                                evi_scor = 1                            
                            evidence_scores_[clean_text].append(evi_scor)
                        plain_sentences_.append(clean_text)
                    else:
                        #print('in else of section tag check')
                        if not section_tagged:
                            print('not section tag')
                        if section_tagged!='REF':
                            #print('in else , if section tag not REF')
                            evidence_scores_[clean_text].append(1)
                            plain_sentences_.append(clean_text)
                        #else:
                        #    print("sentence from reference")
            except:
                pass

            #plain_sentences_.append(clean_text)
    #     calculate average evidence scores
    for each_sentence, scores in evidence_scores_.items():
        average_score = mean(scores)
        average_evidence_scores__[each_sentence] = average_score

    return section_tags_, average_evidence_scores__, plain_sentences_, uniprot_set_, efo_set_, abs_full


# generate dictionary for matches and co-occurances, section and other scores
def generate_interested_sentences_in_json_format(final_sentences, section_tags, match_gp_ds_cd, co_occurance_gp_ds,
                                                 co_occurance_gp_cd, co_occurance_ds_cd):
    interested_sentences = []
    for each_sentence, tags in final_sentences.items():
        minidict = {}

        minidict['text'] = each_sentence

        if section_tags[each_sentence]:
            minidict['section'] = list(section_tags[each_sentence])[0]
        else:
            minidict['section'] = 'Other'
        
        all_matches = match_gp_ds_cd[each_sentence]
        if all_matches:
            #print(all_matches)
            # current assumption is that all_matches element has same sectionStart and sectionEnd if they are there
            if 'sectionStart' in all_matches[0] and 'sectionEnd' in all_matches[0]:
                #print('sectionStart end check')
                minidict['sectionStart'] = all_matches[0]['sectionStart']
                minidict['sectionEnd'] = all_matches[0]['sectionEnd']
                matches = []
                for l in all_matches:
                    l.pop('sectionStart', None)
                    l.pop('sectionEnd', None)
                    matches.append(l)
                minidict['matches'] = matches
            else:
                #print('No section start or end tag')
                minidict['matches'] = all_matches
            
            #print('after adding matches')
        all_co_occurances = co_occurance_gp_ds[each_sentence] + co_occurance_gp_cd[each_sentence] + co_occurance_ds_cd[
            each_sentence]

        if all_co_occurances:
            minidict['co-occurrence'] = all_co_occurances
        if all_co_occurances or all_matches:
            #print('all_co_occurances or all_match')
            interested_sentences.append(minidict)
    return interested_sentences


def get_pmid(lookup_file, pmcid):
    #print('in getpmid')
    
    chunk = pd.read_csv(lookup_file, sep='\t')
    #print("colukmn names")
    #print(chunk.columns)
    #print(chunk.head)
    chunk = chunk.rename(columns=lambda x: x.strip())
    #print(chunk.columns)
    chunk['PMID'] = chunk['PMID'].str.strip()
    chunk['FT_ID'] = chunk['FT_ID'].str.strip()

    chunk.replace('null', '0', inplace=True)
    chunk['PMID'] = chunk['PMID'].fillna(0)
    chunk['PMID'] = chunk['PMID'].astype(int)
    chunk['PMID'] = chunk['PMID'].astype(str)
    #print('after astype')
    '''
    try:
        chunk['FT_ID'] = chunk['FT_ID'].fillna(0)
        chunk['FT_ID'] = chunk['FT_ID'].astype(int)
        chunk['FT_ID'] = chunk['FT_ID'].astype(str)
    except:
        print('Error in readng lookup file')
    '''
    pmid = ''
    if pmcid != '':
        try:
            pub_date = chunk[chunk['FT_ID']==pmcid]
            if pub_date.shape[0]>0:
                pmid = pub_date.iloc[0]['PMID']
                print(pmid)
            else:
                print('pmcid pmid not found')
        except:
            print('Error in get_pmid')
    else:
        print('Empty PMCID')
    return pmid

def get_publication_date(lookup_file, pmcid='', pmid=''):
    #print('In get Publication : ' + pmcid + ', ' + pmid)
    chunksize = 10 ** 6
    pubdate = '1900-01-01'
    #print('before lookup file read')
    chunk = pd.read_csv(lookup_file, sep='\t', skipinitialspace=True)
    chunk = chunk.rename(columns=lambda x: x.strip())
    chunk['FT_ID'] = chunk['FT_ID'].str.strip()
    chunk['PUB_DATE'] = chunk['PUB_DATE'].str.strip()
    #print('after reading lookup file')
    #print(chunk.columns.values)
    #print(chunk.head())
    try:
        chunk['PMID'] = chunk['PMID'].fillna(0)
        #print(chunk['PMID'].isnull())
        #chunk['PMID'] = chunk['PMID'].astype(int)
        chunk['PMID'] = chunk['PMID'].astype(str)
        #print('pmid is in str')
        if pmcid != '':
            #print('pmcid not empty')
            pub_date = chunk[chunk['FT_ID'].str.strip()==pmcid]
            if pub_date.shape[0]>0:
                #print("Pub Date:")
                #print(pub_date.iloc[0]['PUB_DATE'])
                pubdate = datetime.datetime.strptime(pub_date.iloc[0]['PUB_DATE'].strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
                #print(pubdate)
            else:
                print('pmcid pubdate not found')
        elif pmid != '':
            #print('pmid not empty')
            pub_date = chunk[chunk['PMID'].str.strip()==pmid]
            if pub_date.shape[0]>0:
                pubdate = datetime.datetime.strptime(pub_date.iloc[0]['PUB_DATE'].strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
                #print(pubdate)
            else:
                print('ERROR : pmid pubdate not found ' + pmid)
        else:
            sys.stderr.write('PMID and PMCID both empty. Please check get_pmid_pmcid code')
    except Exception as e:
        #print('error in pub date')
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        #print(message)
        #print(e.message)
    return pubdate


def get_publication_date_abstract(soup):
    
    date_year_1 = soup.find('pubmedpubdate')
    date_year_2 = soup.find('datecompleted')
    date_year_3 = soup.find('pubdate')
    year = '1900'
    month = '01'
    day = '01'
    
    if date_year_1:
        try:
            year = date_year_1.year.text
        except:
            year = ''
        try:    
            month = date_year_1.month.text
        except:
            month = ''
        try:
            day = date_year_1.day.text
        except:
            day = ''
    elif date_year_2:
        try:
            year = date_year_2.year.text
        except:
            year = ''
        try:    
            month = date_year_2.month.text
        except:
            month = ''
        try:
            day = date_year_2.day.text
        except:
            day = ''
    elif date_year_3:
        try:
            year = date_year_3.year.text
        except:
            year = ''
        try:    
            month = date_year_3.month.text
        except:
            month = ''
        try:
            day = date_year_3.day.text
        except:
            day = ''
    else:
        year = ''
        month = ''
        day = ''  
   
    pub_date = year+'-'+month+'-'+day 
    
    return pub_date

def generate_final_json(soup, section_tags, og_set, tagged_sentences, match_gp_ds_cd, co_occurance_gp_ds,
                        co_occurance_gp_cd, co_occurance_ds_cd, lookup_file, document_flag):
    json_generated = {}
    #print('In final JSON')
    pmcid=''
    pmid=''
    pprid=''

    pubdate = '1900-01-01'
    if document_flag == 'f':
        try:
            article_id_list = soup.find_all('article-id')
            if article_id_list:
                for article_id in article_id_list:
                    if article_id.has_attr('pub-id-type') and article_id.get('pub-id-type')=='pmid':
                        pmid = article_id.text
                        #print('pmid:' + pmid)
                        #print(article_id)
                    if article_id.has_attr('pub-id-type') and article_id.get('pub-id-type')=='pmcid':
                        pmcid = article_id.text
                        #print('pmcid:' + pmcid)
                        #print(article_id)
                        #print(type(article_id))
        except Exception as e:
            print("No article-id tag")
            print(e)
        if pmcid != None:
            pmcid = 'PMC' + pmcid
            if pmid == '':
                #print('before calling getpmid')
                pmid = get_pmid(lookup_file, pmcid)
                #print('pmid for ' + pmcid + ' from get_pmid ' + pmid)
        else:
            pmcid = ''
    elif document_flag == 'a':
        pmid = soup.find('ext_id') #.text
        if pmid != None:
            if 'PPR' in pmid:
                pprid = pmid.text
            else:
                pmid = pmid.text
        else:
            #print('pmid not found')
            #print(soup)
            pmid = ''
            #sys.exit(1)
    if pmid == '0':
        pmid = ''
    if pmcid=='' and pmid=='' and pprid=='':
        print('pmcid, pmid and pprid empty, hence skipping this file.')
    else:
        json_generated['pmcid'] = pmcid
        json_generated['pmid'] = pmid
        json_generated['pprid'] = pprid
        #if document_flag == 'f':
        try:
            if document_flag == 'f':
                pubdate = get_publication_date(lookup_file, pmcid=json_generated['pmcid'], pmid=json_generated['pmid'])
                #print('after pubdate')
            if pubdate=='1900-01-01' and document_flag == 'a':
                pubdate_element = soup.find('pub_date')
                if pubdate_element!= None and pubdate_element.text:
                    pubdate=datetime.datetime.fromisoformat(pubdate_element.text).strftime('%Y-%m-%d') #get_publication_date_abstract(soup)
                #print('pubdate:' + pubdate)
            json_generated['pubDate'] = pubdate
            json_generated['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
            json_generated['organisms'] = list(og_set)
    
            interested_sentences = generate_interested_sentences_in_json_format(tagged_sentences, section_tags, match_gp_ds_cd, co_occurance_gp_ds, co_occurance_gp_cd,co_occurance_ds_cd)
            json_generated['sentences'] = interested_sentences
        except Exception as e:
            print("error in pub date or generate_interested_sentences_in_json_format")
            print(e)
    #print('returning json_generated')
    return json_generated



def process_each_file_in_job_per_article(each_file_path, lookup_file, document_flag):

    #getfileblocks extract articles from the batch file.
    files_list = getfileblocks(each_file_path, document_flag)
    #print(files_list)
    for each_file in tqdm(files_list):
        try:
            xml_soup = BeautifulSoup(each_file, 'lxml')
            section_tag_sents, average_evidence_scores_sents, plain_sentences, uniprot_set, efo_set, absfull = extract_sentence_level_details(
                xml_soup, document_flag)
            #print(plain_sentences)
            ml_annotations = get_ml_tags(plain_sentences,missing_list)
            #print('ml_annotation')
            #print(ml_annotations)
            mltag_sentences, ml_gp_set, ml_cd_set, ml_og_set = get_only_ml_tagged_sentences(plain_sentences, ml_annotations, missing_list)
            # sent offset
            ml_gp_ds, ml_gp_cd, ml_ds_cd = get_sentences_offset_per_cooccurance(mltag_sentences)
            #print("\n\nafter sent offset")
            # cooccurance evidence - gene/protein and disease
            ml_co_occurance_gp_ds = get_cooccurance_evidence(average_evidence_scores_sents, ml_gp_ds, tag_type_1='GP', tag_type_2='DS')
            #print("co-occure gp_ds")
            # cooccurance evidence - gene/protein and chemical
            ml_co_occurance_gp_cd = get_cooccurance_evidence(average_evidence_scores_sents, ml_gp_cd, tag_type_1='GP', tag_type_2='CD')
            #print("co-occure gp_cd")
            # cooccurance evidence - disease and chemical
            ml_co_occurance_ds_cd = get_cooccurance_evidence(average_evidence_scores_sents, ml_ds_cd, tag_type_1='DS', tag_type_2='CD')
            #print("co-occure ds_cd")
            # cooccurance of gene/protein, disese and chemical entities
            ml_match_gp_ds_cd = get_sentences_matches_tags(mltag_sentences, absfull)
            #print('match gp-ds-cd')
            try:
                ml_json = generate_final_json(xml_soup, section_tag_sents, ml_og_set, mltag_sentences, ml_match_gp_ds_cd,
                                          ml_co_occurance_gp_ds, ml_co_occurance_gp_cd, ml_co_occurance_ds_cd, lookup_file, document_flag)
            except Exception as e:
                print("Exception in generate final json :" + each_file_path)
                print(e)
            #print("after final json")
            if len(ml_json.keys())>0:
                # save ml json
                with open(ml_result_path + 'NMP_' + each_file_path.split('/')[-1][:-3] + 'jsonl', 'at',
                          encoding='utf8') as json_file:
                    json.dump(ml_json, json_file, ensure_ascii=False)
                    json_file.write('\n')
            else:
                print("no json")
        except:
            print('error processing, so skipping file')
            #sys.exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This script will process patch files to extract GP DS CDs in job folders on OTAR FullTextLoadings')
    parser.add_argument("-f", "--file", nargs=1, required=True, help="OTAR New Pipeline GP DS CD extractor to Jsonl format", metavar="PATH")
    parser.add_argument("-o", "--out", nargs=1, required=True, help="ML output folder", metavar="PATH")
    parser.add_argument("-m", "--model", nargs=1, required=True, help="ML model path", metavar="PATH")
    parser.add_argument("-l", "--lookup", nargs=1, required=True, help="TSV file with PMID, PMCID, Publication date etc. information", metavar="PATH")
    parser.add_argument("-d", "--document", nargs=1, required=True, help="Document Type, f for Full text and a for abstracts", metavar="PATH")
    
    args = parser.parse_args()
    ml_result_path = args.out[0]
    pathlib.Path(ml_result_path).mkdir(parents=True, exist_ok=True)

    MODEL_PATH = args.model[0]
    #print('params are fine')

    # path to the file that has model parameters
    params_path = MODEL_PATH + "params.pickle"
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    params['max_ner_token_len'] = -1
    params['max_bert_token_len'] = -1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load ner association and relation models
    ml_model = MLModel()
    #relation_model1 = en_relationv01.load()
    #relation_model2 = en_ner_europepmc_md.load()
    json_file_handle = open(ml_result_path + 'NMP_' + args.file[0].split('/')[-1][:-3] + 'jsonl', 'w', encoding='utf8')
    json_file_handle.close()
                      
    process_each_file_in_job_per_article(args.file[0], args.lookup[0], args.document[0])
    print(args.file[0] + ' : NER finished!')
