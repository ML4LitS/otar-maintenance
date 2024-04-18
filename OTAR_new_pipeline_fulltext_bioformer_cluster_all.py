import io
import sys
import sys
from bs4 import BeautifulSoup
from collections import defaultdict
import argparse
from fuzzywuzzy import fuzz
import re
from tqdm import tqdm
import pathlib
import json

import os
import unicodedata
import datetime
from statistics import mean
import numpy as np
import itertools
import re
import pandas as pd

from optimum.pipelines import pipeline
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig


import onnxruntime as ort
# Set environment variables (adjust numbers according to your Slurm resource allocation)
#os.environ["OMP_NUM_THREADS"] = "4"  # Adjust as per your Slurm --cpus-per-task
#os.environ["ORT_NUM_THREADS"] = "4"  # Similar adjustment


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


def extract_sentence_level_details(soup, document_flag):
    plain_sentences_ = []
    section_tags_ = defaultdict(set)
    evidence_scores_ = defaultdict(list)
    average_evidence_scores__ = defaultdict(list)
    line_count = 0
    #print('In extract_sentence_level_details')
    # get all the sentences, UPDATE : except for the refernce sentences, change code
    all_sentences = soup.find_all('sent')
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

    return section_tags_, average_evidence_scores__, plain_sentences_, abs_full

def merge_with_same_spans(x_list):
    merged_list = []
    for sublist in x_list:
        if merged_list and merged_list[-1][1] == sublist[0] and merged_list[-1][2] == sublist[2]:
            merged_list[-1][1] = sublist[1]
            merged_list[-1][3] += sublist[3]
        else:
            merged_list.append(sublist)
    
    return merged_list
        
        

def get_ml_tags_sentences(all_sentences_):

    all_sentences = [x for x in all_sentences_ if len(x)>10]
    
    no_sentences = 8
    ML_annotations = defaultdict(list)
    OG_set = set()

    for i in range((len(all_sentences) + no_sentences - 1) // no_sentences ):
        batch_sentences = all_sentences[i * no_sentences:(i + 1) * no_sentences]
        pred = ner_quantized(batch_sentences)
        count = 0
        for all_ent in pred:
            my_sentence = batch_sentences[count]
            if all_ent:
                x_list_=[]
                for ent in all_ent:
                    if my_sentence[ent['start']:ent['end']] in ['19', 'COVID', 'COVID-19']:
                        ent['entity_group'] = 'DS'
                    x_list_.append([ent['start'], ent['end'], ent['entity_group'], my_sentence[ent['start']:ent['end']]])
                    if ent['entity_group'] == 'OG':
                        OG_set.add(my_sentence[ent['start']:ent['end']])
                ML_annotations[my_sentence].extend(merge_with_same_spans(x_list_))
            count = count+1
  
    return ML_annotations, OG_set


def check_tags(tags, tag_type):
    flag= 0
    for sublist in tags:
        if sublist[2] == tag_type:
            flag =1
            break
    
    return flag
    
    
# get only those sentences with relevant pairs
def get_sentences_offset_per_cooccurance(sentences_tags):
    dict_gp_ds = defaultdict(list)
    dict_gp_cd = defaultdict(list)
    dict_ds_cd = defaultdict(list)

    for sentence, tags in sentences_tags.items():
        if len(tags) > 1:  # only if more than 1 tag is available
            # check_tags = np.array(tags)
            if check_tags(tags, 'GP') and check_tags(tags, 'DS'):
                dict_gp_ds[sentence] = get_mapped_list_from_annotations(tags)
            if check_tags(tags, 'GP') and check_tags(tags, 'CD'):
                dict_gp_cd[sentence] = get_mapped_list_from_annotations(tags)
            if check_tags(tags, 'DS') and check_tags(tags, 'CD'):
                dict_ds_cd[sentence] = get_mapped_list_from_annotations(tags)

    return dict_gp_ds, dict_gp_cd, dict_ds_cd


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


# fulltext_scores
def assign_scores_to_sections(fulltext_scores_, section_tagged_):
    scores = []
    for key, val in fulltext_scores_.items():
        if key in section_tagged_.lower():
            return val

    return fulltext_scores_['other']


# map annotations in sets of pairs
def get_mapped_list_from_annotations(annotation_list):
    mapped_list = list(itertools.combinations(annotation_list, 2))

    unique_maplist = []
    for each_list in mapped_list:
        if each_list[0][2] != each_list[1][2] and each_list[1][2] != 'OG' and each_list[0][2] != 'OG':
            unique_maplist.append((each_list[0], each_list[1]))

    return unique_maplist


# if not in the right position if the pair and swap them such that always GP is followed by either CD or DS and DS is followed by CD
def swap_positions(cooccurance_list, pos1, pos2):
    cooccurance_list[pos1], cooccurance_list[pos2] = cooccurance_list[pos2], cooccurance_list[pos1]
    return cooccurance_list


def get_pmid(pmcid, chunk):
    pmid = ''
    if pmcid != '':
        try:
            pub_date = chunk[chunk['FT_ID']==pmcid]
            if pub_date.shape[0]>0:
                pmid = pub_date.iloc[0]['PMID']
                # print(pmid)
            else:
                print('pmcid pmid not found')
        except:
            print('Error in get_pmid')
    else:
        print('Empty PMCID')
    return pmid

def get_pmcid(pmid, chunk):
    pmcid = ''
    if pmid!= '':
        try:
            pub_date = chunk[chunk['PMID']==pmid]
            if pub_date.shape[0]>0:
                pmcid = pub_date.iloc[0]['FT_ID']
            else:
                print('pmid pmcid not found')
        except:
            print('Error in get_pmcid')
    else:
        print('Empty PMID')
    return pmcid


def get_publication_date(chunk, pmcid='', pmid=''):
    #print('In get Publication : ' + pmcid + ', ' + pmid)
    chunksize = 10 ** 6
    pubdate = '1900-01-01'
    #print('before lookup file read')
    chunk = chunk.rename(columns=lambda x: x.strip())
    chunk['FT_ID'] = chunk['FT_ID'].str.strip()
    chunk['PUB_DATE'] = chunk['PUB_DATE'].str.strip()
    try:
        chunk['PMID'] = chunk['PMID'].fillna(0)
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
                        co_occurance_gp_cd, co_occurance_ds_cd, document_flag,chunk_):
    json_generated = {}
    #print('In final JSON')
    pmcid=''
    pmid=''
    pprid=None

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
    #  first check whether both pmc and pmids are available 
    #  if available 
        if len(pmcid)>0 and len(pmid)>0:
           # check if pmcid starts with PMC
            if pmcid.startswith('PMC'):
                pmcid = pmcid
            else:
                pmcid = 'PMC' + pmcid


    # if one of them is missing 
        # if pmcid missing 
        if pmcid=='' and len(pmid)>1:
            pmcid = get_pmcid(pmid,chunk_)

        # if pmid is missing
        if pmid == '' and len(pmcid)>0:
            #print('before calling getpmid')
            if pmcid.startswith('PMC'):
                pmcid = pmcid
            else:
                pmcid = 'PMC' + pmcid
                  #print('pmid for ' + pmcid + ' from get_pmid ' + pmid)
            pmid = get_pmid(pmcid,chunk_)

    # if both are missing check it later with pprids

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
     # Final checking        
    if pmid == '0':
        pmid = None
    if pmcid == '':
        pmcid = None
    
    if pmcid==None and pmid==None and pprid==None:
        print('pmcid, pmid and pprid empty, hence skipping this file.')
    else:
        json_generated['pmcid'] = pmcid
        if pmid:
            json_generated['pmid'] = pmid.strip()
        else:
            json_generated['pmid'] = pmid
        json_generated['pprid'] = pprid
        #if document_flag == 'f':
        try:
            if document_flag == 'f':
                pubdate = get_publication_date(chunk_, pmcid=json_generated['pmcid'], pmid=json_generated['pmid'])
                #print('after pubdate')
            if pubdate=='1900-01-01' and document_flag == 'a':
                pubdate_element = soup.find('pub_date')
                if pubdate_element!= None and pubdate_element.text:
                    pubdate=datetime.datetime.fromisoformat(pubdate_element.text).strftime('%Y-%m-%d') #get_publication_date_abstract(soup)
                #print('pubdate:' + pubdate)
            if pubdate=='1900-01-01':
                json_generated['pubDate'] = None
            else:
                json_generated['pubDate'] = pubdate
                
            json_generated['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
            
            if og_set:
                json_generated['organisms'] = list(og_set)
            else:
                json_generated['organisms'] = None
    
            interested_sentences = generate_interested_sentences_in_json_format(tagged_sentences, section_tags, match_gp_ds_cd, co_occurance_gp_ds, co_occurance_gp_cd,co_occurance_ds_cd)
            
            if interested_sentences:
                json_generated['sentences'] = interested_sentences
            else:
                json_generated['sentences'] = None
        
        except Exception as e:
            print("error in pub date or generate_interested_sentences_in_json_format")
            print(e)
    #print('returning json_generated')
    return json_generated

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




def process_each_file_in_job_per_article(each_file_path, lookup_file, document_flag):

    #getfileblocks extract articles from the batch file.
    files_list = getfileblocks(each_file_path, document_flag)
    #print(files_list)
    chunk_ = pd.read_csv(lookup_file, sep='\t', skipinitialspace=True,  dtype={
                 'PMID ': str,
                 'FT_ID ': str,
                    'PUBYEAR ': str,
                    'PUB_DATE ': str,
                    'RESEARCH_ARTICLE_YES_NO ': str,
                    'AVAILABILITY_STATUS ':str,
                    'FIRST_PUBLISH_DATE':str
                    })

    chunk_ = chunk_.rename(columns=lambda x: x.strip())
    chunk_['PMID'] = chunk_['PMID'].str.strip()
    chunk_['FT_ID'] = chunk_['FT_ID'].str.strip()

    chunk_.replace('null', '0', inplace=True)
    chunk_['PMID'] = chunk_['PMID'].fillna(0)

   
    for each_file in tqdm(files_list):
        try:
            xml_soup = BeautifulSoup(each_file, 'lxml')
            section_tag_sents, average_evidence_scores_sents, plain_sentences, absfull = extract_sentence_level_details(
                xml_soup, document_flag)
            #print(plain_sentences)
            mltag_sentences, ml_og_set = get_ml_tags_sentences(plain_sentences)
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
                                          ml_co_occurance_gp_ds, ml_co_occurance_gp_cd, ml_co_occurance_ds_cd, document_flag,chunk_)
            except Exception as e:
                print("Exception in generating final json :" + each_file_path)
                print(e)
            #print("after final json")
            if len(ml_json.keys())>0:
                # save ml json
                with open(ml_result_path + 'NMP_' + each_file_path.split('/')[-1][:-3] + 'jsonl', 'at',
                          encoding='utf8') as json_file:
                            json_data = json.dumps(ml_json, indent=None, ensure_ascii=False, default=lambda x: "null")
                            json_file.write(json_data + "\n")
                    # json.dump(ml_json, json_file, ensure_ascii=False)
                    # json_file.write('\n')
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

    
    # model_path_quantised = '/hps/software/users/literature/textmining/test_pipeline/ml_filter_pipeline/ml_fp_filter/quantised'
#    sess_options = ort.SessionOptions()
#    sess_options.intra_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    sess_options = ort.SessionOptions()
    # Check if SLURM_JOB_ID is set and not empty
    if os.getenv("SLURM_JOB_ID"):
        slurm_cpus_on_node = os.getenv("SLURM_CPUS_ON_NODE")
        if slurm_cpus_on_node:
            slurm_cpus_on_node = int(slurm_cpus_on_node)  # Convert to integer
            sess_options.intra_op_num_threads = slurm_cpus_on_node
            sess_options.inter_op_num_threads = slurm_cpus_on_node

    model_path_quantised = args.model[0]
    #print('params are fine')
    model_quantized = ORTModelForTokenClassification.from_pretrained(model_path_quantised, file_name="model_quantized.onnx", sess_options=sess_options)
    tokenizer_quantized = AutoTokenizer.from_pretrained(model_path_quantised, model_max_length=512, batch_size=4, truncation=True)

    ner_quantized = pipeline("token-classification", model=model_quantized, tokenizer=tokenizer_quantized, aggregation_strategy="first")


   
    json_file_handle = open(ml_result_path + 'NMP_' + args.file[0].split('/')[-1][:-3] + 'jsonl', 'w', encoding='utf8')
    json_file_handle.close()
                      
    process_each_file_in_job_per_article(args.file[0], args.lookup[0], args.document[0])
    print(args.file[0] + ' : NER finished!')

    
    # python OTAR_new_pipeline_fulltext_bioformer_cluster_all.py -f $sectionpath/patch-$TIMESTAMP-$file_index.xml -o $mlpath/ -m $model -l $summarypath/otar-$TIMESTAMP.tsv -d f
