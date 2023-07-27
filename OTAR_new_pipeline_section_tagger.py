########## This code adds section tags to the annotated files. #######

from bs4 import BeautifulSoup
import lxml
from collections import defaultdict
from tqdm import tqdm
import random
import sys, io, re, os
import argparse

titleMapsBody = {
    'INTRO': ['introduction', 'background', 'related literature', 'literature review', 'objective', 'aim ', 'purpose of this study', 'study (purpose|aim|aims)', '(\d)+\. (purpose|aims|aim)', '(aims|aim|purpose) of the study', '(the|drug|systematic|book) review', 'review of literature', 'related work', 'recent advance'],
    'METHODS': ['supplement', 'methods and materials', 'method', 'material', 'experimental procedure', 'implementation', 'methodology', 'treatment', 'statistical analysis', "experimental", '(\d)+\. experimental$', 'experimental (section|evaluation|design|approach|protocol|setting|set up|investigation|detail|part|pespective|tool)', "the study", '(\d)+\. the study$', "protocol", "protocols", 'study protocol', 'construction and content', 'experiment (\d)+', '^experiments$', 'analysis', 'utility', 'design', '(\d)+\. theory$', "theory", 'theory and ', 'theory of '],
    'RESULTS': ['result', 'finding', 'diagnosis'],
    'DISCUSS': ['discussion', 'management of', '(\d)+\. management', 'safety and tolerability', 'limitations', 'perspective', 'commentary', '(\d)+\. comment'],
    'CONCL': ['conclusion', 'key message', 'future', 'summary', 'recommendation', 'implications for clinical practice','concluding remark'],
    'CASE': ['case study report', 'case report', 'case presentation', 'case description', 'case (\d)+', '(\d)+\. case', 'case summary', 'case history'],
    'ACK_FUND': ['funding', 'acknowledgement', 'acknowledgment', 'financial disclosure'],
    'AUTH_CONT': ['author contribution', 'authors\' contribution', 'author\'s contribution'],
    'COMP_INT': ['competing interest', 'conflict of interest', 'conflicts of interest', 'disclosure', 'decleration'],
    'ABBR': ['abbreviation'],
    'SUPPL': ['supplemental data', 'supplementary file', 'supplemental file', 'supplementary data', 'supplementary figure', 'supplemental figure', 'supporting information', 'supplemental file', 'supplemental material', 'supplementary material', 'supplement material', 'additional data files', 'supplemental information', 'supplementary information', 'supplemental information', 'supporting information', 'supplemental table', 'supplementary table', 'supplement table', 'supplementary material', 'supplemental material', 'supplement material', 'supplementary video']
}

titleExactMapsBody = {
    'INTRO': ["aim", "aims", "purpose", "purposes", "purpose/aim", "purpose of study", "review", "reviews", "minireview"],
    'METHODS': ["experimental", "the study", "protocol", "protocols"],
    'DISCUSS': ["management", "comment", "comments"],
    'CASE': ["case", "cases"]
}

titleMapsBack = {
    'REF': ['reference', 'literature cited', 'references', 'bibliography'],
    'ACK_FUND': ['funding', 'acknowledgement', 'acknowledgment', 'aknowledgement', 'acknowlegement', 'open access', 'financial support', 'grant', 'author note', 'financial disclosure'],
    'ABBR': ['abbreviation', 'glossary'],
    'COMP_INT': ['competing interest', 'conflict of interest', 'conflicts of interest', 'disclosure', 'decleration', 'conflits', 'interest'],
    'SUPPL': ['supplementary', 'supporting information', 'supplemental', 'web extra material'],
    'APPENDIX': ['appendix', 'appendices'],
    'AUTH_CONT': ['author', 'contribution']
}


def createSecTag(soup, secType):
    secTag = soup.new_tag('SecTag')
    secTag['type'] = secType
    return secTag


def titlePartialMatch(title, secFlag):
    matchKeys = []
    if secFlag == 'body':
        for key in titleMapsBody.keys():
            if any(re.search(pattern,title.lower()) for pattern in titleMapsBody[key]):
                #check for the exact match cases
                matchKeys.append(key)
    if secFlag == 'back':
        for key in titleMapsBack.keys():
            if any(re.search(pattern,title.lower()) for pattern in titleMapsBack[key]):
                matchKeys.append(key)
    if len(matchKeys)>0:
        return ','.join(matchKeys)
    else:
        return None

def titleExactMatch(title):
    for key in titleMapsBody.keys():
        if any(pattern==title.lower() for pattern in titleMapsBody[key]):
            #check for the exact match cases
            return key
    return None


#add SecTag with appropriate title
def section_tag(soup):
    # Add Figure section
    for fig in soup.find_all(['fig'], recursive=True):
        if fig.find_all(['fig'], recursive=True):
            # This fig tag is a parent fig tag. We do not want to wrap it with the section tag. so we continue.
            continue
        else:
            fig_tag = createSecTag(soup, 'FIG')
            fig.wrap(fig_tag)
    # Add Table section
    for fig in soup.find_all(['table-wrap'], recursive=True):
        if fig.find_all(['table-wrap'], recursive=True):
            # This fig tag is a parent fig tag. We do not want to wrap it with the section tag. so we continue.
            continue
        else:
            fig_tag = createSecTag(soup, 'TABLE')
            fig.wrap(fig_tag)
    # get front section
    if soup.front:
        if soup.front.abstract:
            secAbs = createSecTag(soup, 'ABSTRACT')
            soup.front.abstract.wrap(secAbs)
        if soup.front.find('kwd-group'):
            secKwd = createSecTag(soup, 'KEYWORD')
            soup.front.find('kwd-group').wrap(secKwd)
    # get sec tags from body
    # find sec tag and their titles, as sectag type using a dictionary mapping
    secFlag = 'body'
    #print(secFlag)
    if soup.body:
        for sec in soup.body.find_all('sec', recursive=False):
            if sec.title:
                #print(sec.title.text)
                mappedTitle = titleExactMatch(sec.title.text)
                if mappedTitle is None:
                    mappedTitle = titlePartialMatch(sec.title.text, secFlag)
                #print(mappedTitle)
                if mappedTitle:
                    secBody = createSecTag(soup, mappedTitle)
                    sec.wrap(secBody)
    
    # get back sections
    # find sec tag and their titles, as sectag type using a dictionary mapping
    secFlag = 'back'
    #print(secFlag)
    if soup.back:
        # apply the title mapping to sec and the special cases, like app-group, ack, ref-list
        for sec in soup.back.find_all(['sec', 'ref-list', 'app-group', 'ack', 'glossary', 'notes', 'fn-group'], recursive=False):
            if sec.title:
                #print(sec.title.text)
                mappedTitle = titlePartialMatch(sec.title.text, secFlag)
                #print(mappedTitle)
                if mappedTitle:
                    secBack = createSecTag(soup, mappedTitle)
                    sec.wrap(secBack)
            else:
                if sec.name=='ref-list':
                    secRef = createSecTag(soup, 'REF')
                    sec.wrap(secRef)


def getfileblocks(file_path):
    sub_file_blocks = []
    start_str1 = '<articles><article '
    start_str2 = '<article '
    try:
        with io.open(file_path, 'r', encoding='utf8') as fh:
            for line in fh:
                if line.startswith(start_str1) or line.startswith(start_str2):
                    sub_file_blocks.append(line.replace('^<articles>', ''))
                else:
                    sub_file_blocks[-1] += line.strip().replace('</articles>$', '') 
    except:
        #print('error processing, skipping file : ' + file_path)
        with io.open(file_path, 'r', encoding='ISO-8859-1') as fh:
            for line in fh:
                if line.startswith(start_str1) or line.startswith(start_str2):
                    sub_file_blocks.append(line.replace('^<articles>', ''))
                else:
                    sub_file_blocks[-1] += line.strip().replace('</articles>$', '')
    return sub_file_blocks


def process_each_file(filename, outfolder):
    #print('Process Each File')
    #print(filename)
    #print(outfolder)
    files_list = getfileblocks(filename)
    #print("Files _list size : " + str(len(files_list)))
    out_file = os.path.splitext(os.path.basename(filename))[0]
    #print(out_file)
    count = 0
    with open(outfolder + out_file + ".xml", 'w') as fa:
        pass
    for each_file in tqdm(files_list):
        #print("count : " + str(count))
        try:
            count = count + 1
            #print('\n\n\n')
            each_file = each_file.replace('<body>', '<orig_body>')
            each_file = each_file.replace('<body ', '<orig_body ')
            each_file = each_file.replace('</body>', '</orig_body>')
            #print(each_file)
            xml_soup = BeautifulSoup(each_file, 'lxml')
            xml_soup.html.unwrap()
            xml_soup.body.unwrap()
            if xml_soup.find('orig_body'):
                #print("orig_body found")
                xml_soup.find('orig_body').name = 'body'
            else:
                continue
                #print("orig BODY not found, WHYYYY")
            section_tag(xml_soup)
            with open(outfolder + out_file + ".xml", 'a') as fa:
                fa.write(str(xml_soup) + '\n')
        except Exception as e:
            print('error processing, parse error' + str(e))
            print(each_file)
            sys.exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This script will process sentence annotated xml files, chunk them, maximum of 200 articles in a chunk and add section tag')
    parser.add_argument("-f", "--file", nargs=1, required=True, help="Sentence annotated xml file", metavar="PATH")
    parser.add_argument("-o", "--out", nargs=1, required=True, help="Output folder", metavar="PATH")
    
    args = parser.parse_args()
    #print(args.out[0])
    process_each_file(args.file[0], args.out[0])
    print(args.file[0] + ": section tagging finished")
