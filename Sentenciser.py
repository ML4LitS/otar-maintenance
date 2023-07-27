# This is a XML aware sentenciser for JATS DTD. Sentence splitting rules are based on SSSplit.
# Developed by Shyamasree Saha, 2021

import argparse
import sys
import regex as re
import io
import gzip
from bs4 import BeautifulSoup
import lxml
from collections import defaultdict
from tqdm import tqdm

# sys.setrecursionlimit(2500)
# print(sys.getrecursionlimit())
## defining all the patterns
bad_white_space = "(\t|\r|\n)"
space_btw_tags = "> <"
single_fig_graphic = "<FIGURE/>|<graphic.*?/>"  # replace with ""
fig_stop = "\\.<fig "  # replace with ". <fig "
table_wrap_stop = "\\.<table-wrap "  # replace with ". <table-wrap "
allowedAttrChars = "[\\w-_\u2013\\.\\(\\)\\[\\]]"
refSCIgeneral = "(?:(?:(?:\\d+[,\u2013])*\\d+)?(?:</italic>)?<REF(?:(?: ID=\"(?:\\w-)?\\w+?(?: \\w+?)*\")|(?: " \
                "REFID=\"\\w+?\")|(?: text=\"(?:refs?\\.(?: )?)?(?:(?:\\d+[,�\u2013\u002D])*\\d+)?\\w*?\")|(?: " \
                "TYPE=\"\\w+?\"))*(?: ?/>|>(?:(?:refs?\\.(?: )?)?(?:(?:(?:\\d+(?:<italic>|<i>)?\\w?(?:</italic>|</i>)?)|(?:(" \
                "?:<italic>|<i>)?\\w(?:</italic>|</i>)?[-,\u2013])|(?:(?:<italic>|<i>)?\\w(?:</italic>|</i>)?[-,\u2013]))*(?:(?:\\d+(" \
                "?:<italic>|<i>)?\\w?(?:</italic>|</i>)?)|(?:(?:<italic>|<i>)?\\w?(?:</italic>|</i>)?)))?</REF>))|(?:<xref(?: " \
                "id=\"(?:\\w-)?\\w+?(?: \\w+?)*\")?(?: ref-type=\"(" \
                "?:aff|app|author-notes|bibr|boxed-text|chem|contrib|corresp|disp-formula|fig|fn|kwd|list|plate|ref" \
                "|scheme|sec|statement|supplementary-material|table|table-fn|other)\")?(?: rid=\"(?:[" \
                "-\u2013\\w])*\")?>(?:(?:<sup>)?[-\u2013,\\d]+(?:</sup>))</xref>))"
# refFootnote = "<SUP TYPE=\"FOOTNOTE_MARKER\" ID=\"[^\"]+?\"/>"
atLeastOneRefSCI = "(" + refSCIgeneral + "+" + "\\)?|\\))"  # there may be a bracket after the
# reference
capturePunctuation = "(\\.|\\?|(?<!\\(\\w{1,15})\\!)"
abbreviations = '((\\(|\\[| |>|^)(al|Am|Angew|approx|Biochim|Biophys|ca|cf|Chem|Co|conc|Dr|Drs|Corp|Ed|no|No|Sci|Nat' \
                '|Rep|e\\.g|eg|p\\.p\\.m|Engl|eq|eqns?|exp|Rs|Figs?|Labs?|Dr|etc|Calc|i\\.e|ie|Inc|Int|Lett|Ltd|p|p\\.a' \
                '|Phys|Prof|prot|refs?|Rev|sect|st|vs|(?<!(?:</SB>|(?:\\d ?(<italic>?))|<italic>))(?:(?:(?:[A-Z]|[' \
                'a-z])?\\.) ?[a-z])|(?<!(?:</SB>|(?:\\d ?(<italic>?))|<italic>))(?:(?:(?:[A-Z]|[a-z])\\.)? ?(?:(?:[A-Z]|[' \
                'a-z])\\.)? ?[A-Z])|(?<!(?:</SB>|(?:\\d ?(<italic>?))|<italic>))(?:(?:(?:[A-Z]|[a-z])\\.)? ?(?:(?:[A-Z]|[' \
                'a-z])\\.) ?[a-z])|(?<=(?:(?<!(?:</SB>|(?:\\d)))(?:(?: |\\(|^)<italic>)))(?: ?\\w{1,10})| ?\\. ?\\. ?|\\())'
# moving all references inside the punctuation so that the sentence splitting is easier
refSentence = "(.*?)" + capturePunctuation + atLeastOneRefSCI + "(\\s?(?:</P>)?)"
abbrevs = ".*?(" + abbreviations + "(?:|</i>|</italic>)?)" + "$"
capitals = "[A-Z0-9]"  # caps and numbers may begin a sentence
punctuation = "(?:\\.|\\?|(?<!\\(\\w{1,15})\\!)"  # .?!     "(?:\\.|\\?|(?<!\\(\\w{1,15}))\\!|\\:(?=<list.+>))"
optPunctuation = punctuation + "??"
endEquation = "</EQN>"
# String endPara1 = "(</P>|</ABSTRACT>)"
endPara = "(</p>|</list>)"
beginPara = "<p>"
# notFollowFigTab = "(?!(<fig)|(<table-wrap))";
optStartQuote = "['\"\u201C]?"
optCloseQuote = "['\"\u201D]?"
optReferenceSCI = refSCIgeneral + "*"
beginFirstSentence = "^"
endLastSentence = "$"
openHeader = "(<HEADER/>|<label>)"
wholeHeader = "((<BODY>)?(<DIV( DEPTH=\"\\d+\")?>)?(<HEADER( HEADER_MARKER=\"" + allowedAttrChars + \
              "+?\")?>.*?</HEADER>|<HEADER/>)|<TITLE>)"
optOpenHeader = openHeader + "?"

eqn = "<EQN( ID=\"" + allowedAttrChars + "+?\")?( TYPE=\"" + allowedAttrChars + "+?\")?>"
xref = "<xref( .+)?>"
listTag = "<list.+?>"
# manyStartTags = "(" + eqn + "|" + xref + "|<BODY>|<DIV( DEPTH=\"\d+\")?>|<P>|<B>|<IT>)*"
manyStartTags = "(" + xref + "|" + listTag + "|<p.+?>|<license-p>|<title>|<caption>|<sec.+?>|<disp-quote>|<supplementary-material.+?>|<boxed-text.+?>|<list list-type=\"\\w{2,20}\">|<list-item>|<article-title>|<abstract.+>|<AbstractText.+>|<statement>|<def>|<fig .+>)*"  # shs listTag on 13/10/09, <ABSTRACT>|<abstract.+> on 09/11/09
optEndTags = "(</label>|</boxed-text>|</list>|</list-item>|</statement>|</copyright-statement>)?"
# <list.+?> added to optEndTags1
optEndTags1 = "(</caption>|</boxed-text>|<list.+?>|</list>|</list-item>|</statement" \
              ">|</inline-supplementary-material>|<inline-supplementary-material/>|</related-article>|</related" \
              "-object>|</address>|</alternatives>|</array>|</boxed-text>|</chem-struct-wrap>|</fig>|</fig-group" \
              ">|<graphic.+?/>|</media>|</preformat>|</supplementary-material>|</table-wrap-foot>|</table-wrap" \
              ">|</table-wrap-group>|</disp-formula>|</disp-formula-group>|</element-citation>|</mixed-citation" \
              ">|</nlm-citation>|</bold>|</italic>|</monospace>|</overline>|</overline-start>|</overline-end>|</roman" \
              ">|</sans-serif>|</sc>|</strike>|</underline>|</underline-start>|</underline-end>|</award-id>|</funding" \
              "-source>|</open-access>|</chem-struct>|<inline-formula/>|<inline-graphic/> " \
              "|</private-char>|</def-list>|</list>|</tex-math>|</mml:math>|</abbrev>|</milestone-end>|</milestone" \
              "-start>|</named-content>|</styled-content>|</ack> " \
              "|</disp-quote>|</speech>|</statement>|</verse-group>|</fn>|</target>|</xref>|</ref>|</sub>|</sup>" \
              ">|</def>)*"
# </P> added on 23-11-10 by shs
endTags = "(</licence-p>|</italic>|</sup>|</bold>|</article-title>|</title>|</abstract>|<disp-formula>|</disp-quote>|</supplementary-material>|</boxed-text>|</list>|</list-item>|</statement>)"
manyEndTags = endTags + "*"
# String sentenceTerminator = "(?>" +endPara+ "|"+ endEquation +"|" + "(?<!(\ |>)refs?)"+punctuation+")"
endParaOrEq = "(" + endPara + "|" + endEquation + ") ?"  # shs notFollowedBy added
formatting = "(<B>|<IT>|<i>|<SP>|<italic>|<BOLD>|<bold>|<sup>|<disp-formula>|<named-content.+?>)"
puncNoAbbrv = "(?<!" + abbreviations + "(</italic>)?)" + punctuation + " "
greekLetters = "[\u0370-\u03FF\u1F00-\u1FFF]"
pAttr = "<p.+?>"
sentenceCommencer = "(?>" + beginPara + "|" + pAttr + "|" + "Fig(s)?\\." + "|" + "fig(s)?\\." + "|" + capitals + "|" + \
                    formatting + "|" + "\\[|\\(|" + greekLetters + "|\u007C)"
equationCommencer = "(" + eqn + ".)"
commencer = "(" + sentenceCommencer + "|" + equationCommencer + ")"
noSpaceReqLookahead = manyStartTags + optOpenHeader + optStartQuote + commencer
nocapsParaLookAhead = "( ?<P>)"
startSentence = manyStartTags + optStartQuote + commencer
# For matching the end of the previous sentence
sentenceFigLookbehind = "(?<=(?<!" + abbreviations + punctuation + ")((" + endParaOrEq + ")|(" + puncNoAbbrv + ")|(" \
                        + optPunctuation + optEndTags + endTags + " ?)))"

# for matching the start of a sentence following a header
headerLookahead = "(?=(?:" + manyStartTags + optOpenHeader + optStartQuote + commencer + "))"
# (Headerstuff | ((normal sentence | firstsentence) sentenceContent, (punctuation|endEquation), optionalEndings,
# lookahead))
# String figLookAhead = "";
Figure = "<fig ?.+?</fig>"
tableWrap = "<table-wrap.+?</table-wrap>|<table-wrap-foot.+?</table-wrap-foot>"
title = "((?:<title/>)|(?:<title.+?</title>))"  # "<title.+?</title>"
# String absBodyLookBehind = "((?:<abstract>)|(?:<abstract.+?>)|(?:<body>))"
secLookBehind = "((?:<sec>)|(?:<sec.+?>))?"  # <sec> can be followed by<body>.
supplimentbehind = "<supplementary-material.+?</supplementary-material>"
refList = "<ref-list.+?</ref-list>"
boxed = "(?:<boxed-text.+?>)(?:<caption>)?"


# read xml files
def getfileblocks(file_path, document_flag):
    sub_file_blocks = []
    if document_flag == 'f':
        xml = '<?xml version="1.0" encoding="UTF-8"?>'
        doc_str = '<!DOCTYPE '
        start_str = '<article '
        start_str2 = '<articles><article '
        doc_str_flag = 0
        try:
            with gzip.open(file_path, 'rt', encoding='utf8') as fh:
                for line in fh:
                    if line.startswith(start_str) or line.startswith(start_str2):
                        # sub_file_blocks.append(xml)
                        sub_file_blocks.append(line.replace('^<articles>', ''))
                        doc_str_flag = 0
                    elif line.startswith(doc_str):
                        doc_str_flag = 1
                    else:
                        # print("line:" + line)
                        if doc_str_flag != 1:
                            sub_file_blocks[-1] += line.strip().replace('</articles>$', '')
        except Exception as e:
            print('error processing, using ISO-8859-1 for file : ' + str(len(sub_file_blocks)))
            with gzip.open(file_path, 'rt', encoding='ISO-8859-1') as fh:
                for line in fh:
                    if line.startswith(start_str) or line.startswith(start_str2):
                        sub_file_blocks.append(line.replace('^<articles>', ''))
                        doc_str_flag = 0
                    elif line.startswith(doc_str):
                        doc_str_flag = 1
                    else:
                        if doc_str_flag != 1:
                            sub_file_blocks[-1] += line.strip().replace('</articles>$', '')
    elif document_flag == 'a':
        start_str1 = '<article>'
        start_str2 = '<articles>'
        with gzip.open(file_path, 'rt', encoding='utf8') as fh:
            for line in fh:
                if line.startswith(start_str1):
                    sub_file_blocks.append(line)
                elif line.startswith(start_str2):
                    continue
                else:
                    sub_file_blocks[-1] += line
    else:
        print('ERROR: unknown document type :' + document_flag)

    return sub_file_blocks


def sentence_split(clear_string, sid):
    # clear_string = str(xml_item)
    #print(clear_string)
    # Apply all replace statements
    clear_string = re.sub(bad_white_space, '', clear_string)
    clear_string = re.sub(space_btw_tags, '><', clear_string)
    clear_string = re.sub(single_fig_graphic, '', clear_string)
    clear_string = re.sub(fig_stop, '. <fig ', clear_string)
    clear_string = re.sub(table_wrap_stop, '. <table-wrap ', clear_string)
    clear_string = re.sub("</xref>, <xref", "</xref><xref", clear_string)
    # print(refSentence[740:])
    refSentence_pattern = re.compile(refSentence, flags=0)
    matches = re.finditer(refSentence_pattern, clear_string)
    # print(matches)
    swapped_string = ""
    ref_last = None
    for refm in matches:
        a = refm.group(1)  # sentence
        b = refm.group(2)  # punctuation
        c = refm.group(3)  # reference
        # print("a:" + a + "\n b:" + b + "\n c:" + c + "\n")
        abbrevm = re.search(abbrevs, a)
        if abbrevm:
            swapped_string = swapped_string + a  # don't change order
            swapped_string = swapped_string + b
            swapped_string = swapped_string + c
        else:
            # for a I use match instead of search because match can search through newline etc. Whereas for c,
            # I am looking for a number at the beginning of a sentence, so it does not matter.
            if re.search('[0-9]]$', a) and re.search('^[1-9]]', c) and b == ".":
                # printl("a ends with number and c starts with number, in character by character comparison")
                swapped_string = swapped_string + a  # don't change order
                swapped_string = swapped_string + b
                swapped_string = swapped_string + c
            else:
                # print("order swapping")
                swapped_string = swapped_string + a  # sentence
                swapped_string = swapped_string + c  # reference or bracket
                swapped_string = swapped_string + b + " "  # punctuation
        ref_last = refm
    if ref_last:
        end_bit = clear_string[ref_last.end():]
    else:
        end_bit = clear_string[len(swapped_string):]
    # swapped_string = re.sub("</xref><xref", "</xref>, <xref", swapped_string)
    # print('swapped string')
    # print(swapped_string)
    # print('endbit')
    swapped_string = swapped_string + end_bit
    clear_string = swapped_string
    clear_string = re.sub(">\\.<", ">. <", clear_string)
    # print('\n\n\n Clear String:\n')
    # print(clear_string)
    sentence_pattern = "(" + sentenceFigLookbehind + "(" + secLookBehind + title + ")+" + ")|" + \
                       "(" + sentenceFigLookbehind + "(" + boxed + secLookBehind + title + "?)" + optEndTags1 + ")|" + \
                       "(" + sentenceFigLookbehind + wholeHeader + headerLookahead + ")|" + \
                       "(" + sentenceFigLookbehind + "(((" + tableWrap + ")+(" + Figure + ")+)|((" + Figure + ")+(" + \
                       tableWrap + ")+)|(" + tableWrap + ")+|(" + Figure + ")+)" + optEndTags + ")|" \
                       + "(" + sentenceFigLookbehind + "(((" + supplimentbehind + ")+" + optEndTags + ")|((" + \
                       supplimentbehind + ")+" + secLookBehind + title + ")))|" \
                       + "(" + sentenceFigLookbehind + "(" + refList + ")+" + optEndTags1 + ")|" \
                       + "(((" + sentenceFigLookbehind + "" + startSentence + \
                       ")|" + beginFirstSentence + "|" + beginPara + ")" + \
                       "(.*?)(Fig(s)?\\..+?)*?" + \
                       "(((?<!(" + endEquation + " ?|" + puncNoAbbrv + " ?|" + endPara + " ?))" \
                       + "(?=(?:" + nocapsParaLookAhead + ")))|" + \
                       "((?>" + endEquation + " ?|" + puncNoAbbrv + " ?|" + endPara + " ?)" \
                       + optCloseQuote + optReferenceSCI + manyEndTags + " ?" \
                       + "(?=(?:" + noSpaceReqLookahead + "|" + nocapsParaLookAhead + "|\n| *$)))|" + endLastSentence \
                       + "))"
    somethingFound = 0
    sentences = []
    # print('before finding sentences\n')
    for m in re.finditer(sentence_pattern, clear_string):
        somethingFound = 1
        # print('M TUPLE:')
        # print(m.group(0))
        # print('\n\n')
        sentences.append(m.group(0))
    if somethingFound == 0:
        print('No Sentence found')
    # else:
    #    print(sentences)
    refSentenceRev = re.compile("(.*?)" + atLeastOneRefSCI + capturePunctuation + "(\\s?(?:</p>)?)\\Z")
    count = 0
    newSentences = []
    for s in sentences:
        # print("in sentences: " + str(count) + ": " + s + "\n")
        refmRev = re.match(refSentenceRev, s)
        # print("refSentenceRev:"+refSentenceRev)
        # if sentence finishes with reference + punctuation, swap the two over
        if refmRev:
            # print("ref found")
            a = refmRev.group(1)  # sentence
            b = refmRev.group(2)  # reference or bracket or both
            c = refmRev.group(3)  # punctuation
            d = refmRev.group(4)  # space and/or </P>
            # print("sentence " + str(count) + ": " + s+ "A: " + a + "\n B: " + b + "\n C: "+ c +"\n", s, a, b, c)
            ns = ""
            # if (b.equals(")")) {
            if b.endswith(
                    ")"):  # { // made it endsWith instead of equals on 28/03/11 coz if reference is within bracket we do not want to swap punctuation, it worked, 31,39). is no longer becoming 31,.39) , why square bracket is not considered in bracket?????
                # #println("b ends with )")
                ns = ns + a  # sentence
                ns = ns + b  # bracket
                ns = ns + c  # punctuation
                ns = ns + d  # space
            else:
                # print("b does not end with )")
                if not b.startswith("<"):
                    index = b.index("<")
                    # print("index:"+index)
                    b1 = b[0:index]  # end index exclusive
                    # print("B1:"+b1)
                    # print("b at 0th position"+b.toCharArray()[0])
                    b2 = b[:index]
                    a = a + b1
                    b = b2
                ns = ns + a  # sentence
                ns = ns + c  # punctuation
                ns = ns + b  # reference
                ns = ns + d  # space
            # print("sentence " + str(count) + ": " + s + "A: " + a + "\n B: " + b + "\n C: " + c + "\n", s, a, b, c)
            newSentences.append(ns)
            # print("Sentence found: " + str(count))
        else:
            # print("sentence added s:"+s)
            if s.replace(" +", "") != "":
                newSentences.append(s)
        # print("not if nor else : count="+count)
        count = count + 1
    sentences = newSentences
    # Post-processing sentence array to move XML tags outside the sentences
    # ppSentenceTag = "(<s sid=\"\\d+\">)"
    ppStartTags = "((?:<abstract>|<body>|<abstracttext(?:.+?)>|<sec(?:.+?)>|<div(\\sDEPTH=\"\\d+\")?>|<p>|<p.+>|<disp-quote>|<list.+?>|<list-item>|<abstract.+?>|<statement>|<boxed-text.+?>|<boxed-text.+?><graphic.+?>)*)"
    # added overall brackets and ?: 15/6/09 mal //shs <sec> <abstract.+?>
    ppSentence = "(.+?)"
    ppEndTags = "((?:</abstract>|</body>|</abstract-title>|</sec>|</div>|</p>|</disp-quote>|</list>|</list-item>|<boxed-text.+?><graphic.+?></graphic></boxed-text>|</graphic>|</boxed-text>|</statement>|<list.+?>|<list-item>)*\\s?)\\Z"
    pp = re.compile(ppStartTags + ppSentence + ppEndTags)
    ppHeader = re.compile(".*?<caption.+?", flags=re.I)
    ppTitle = re.compile(".*?<title.*?>|.*?<title>", flags=re.I)
    ppFig = re.compile("<fig\\s?.+?>|<table-wrap.+?>", flags=re.I)
    # ppCap = re.compile("</caption.+?>|</fn.+?>",Pattern.CASE_INSENSITIVE); //shs comment : seems it can't find the tag without .+?
    ppCap = re.compile("</caption.+?>|</fn.+?>", flags=re.I)
    paraAttr = re.compile("<p.+></p>|<p>", flags=re.I)
    suppli = re.compile("<supplementary-material.+?>", flags=re.I)
    # named = re.compile("<named-content.+?>", flags=re.I)
    # boxed_text = re.compile("<boxed-text.+?>", flags=re.I)
    sent_id = sid
    # Vector<Element> allSentences = new Vector<Element>();
    # print("before sentences loop.")
    nsentences = []
    # print('before filling nsentences')
    for s in sentences:
        # print(s)
        ppm = re.search(pp, s)
        if ppm:
            # print("Sentence " + str(sent_id) + " matches the post- processing tags")
            one = ""
            if ppm.group(1):
                one = ppm.group(1)
            two = ppm.group(3)
            three = ""
            if ppm.group(4):
                three = ppm.group(4)
            # print("Group \nSent id is: " + str(sent_id) + " Group 1 is: " + ppm.group(1) + " Group 2 is: " + ppm.group(2) + "Group 3 is: " +ppm.group(3) + "Group 4 is: " + ppm.group(4))
            # print("one:" + str(one) + ", two: " + str(two) + ", three: " + str(three))
            mHead = re.search(ppHeader, s)
            mTitle = re.search(ppTitle, s)
            mfig = re.search(ppFig, s)
            mCap = re.search(ppCap, s)
            mParaAttr = re.search(paraAttr, s)
            mSuppli = re.search(suppli, s)
            # mNamed = re.search(named, s)
            # mBoxed = re.search(boxed_text, s)
            # s.insertChild(titleNodes.get(titleNodes.size()-1).getValue(), 0);
            if not (mHead or mTitle or mfig or mCap or mParaAttr or mSuppli):
                # print('head, title, mfig etc')
                # print(one + "<sent id=\"" + str(sent_id) + "\">" + two + "</sent>" + three)
                nsentences.append(
                    str(one) + "<SENT id=\"" + str(sent_id) + "\"><plain>" + str(two) + "</plain></SENT>" + str(three))
                # print("Sentence " + str(sent_id) + ": " + s)
                # print("after s tag:"+one + "<s sid=\"" + id+ "\">" + two + "</s>" + three);
                # if re.search(ppFig, two):
                #    print("two contains fig")
                # print(two)
                sent_id = sent_id + 1
            else:
                # this is a header, mHead matches
                # print("\n\n\nHeader : "+s+"\n\n ")
                '''
                if mHead:
                    # print("after s tag:" + one + "<sent id=\"" + str(sent_id) + "\">" + two + "</sent>" + three)
                    print("Head matched.")
                if mTitle:
                    # print("after s tag:" + one + "<sent id=\"" + str(sent_id) + "\">" + two + "</sent>" + three)
                    print("Title matched.")
                if mfig:
                    # print("after s tag:" + one + "<sent id=\"" + str(sent_id) + "\">" + two + "</sent>" + three)
                    print("ppFig matched.")
                if mCap:
                    # print("after s tag:" + one + "<sent id=\"" + str(sent_id) + "\">" + two + "</sent>" + three)
                    print("mCap matched.")
                if mParaAttr:
                    # print("after s tag:" + one + "<sent id=\"" + str(sent_id) + "\">" + two + "</sent>" + three)
                    print("ParaAttr matched.")
                if mSuppli:
                    # print("after s tag:" + one + "<sent id=\"" + str(sent_id) + "\">" + two + "</sent>" + three)
                    print("mSuppli matched.")
                '''
                # print('No head, title, mfig etc')
                # print(s)
                nsentences.append(s)  # ("<SENT id=\"" + str(sent_id) + "\">" + two + "</SENT>")

        else:
            print("Sentence " + str(sent_id) + "-- " + str(s) + " -- didn't match!")
    # print('\n\n\n Clear String:\n')
    # print(clear_string)
    # print("N sentences")
    # print(nsentences)
    clear_string = "".join(nsentences)
    clear_string = re.sub("</xref><xref", "</xref>, <xref", clear_string)
    clear_string = re.sub("</SENT> <SENT ", "</SENT><SENT ", clear_string)
    return sent_id, clear_string


# return StringInt(finalbuffer,sent_id)
def create_tag(soup, sid):
    sent_tag = soup.new_tag('SENT')
    sent_tag['id'] = sid
    return sent_tag


def reference_sents(ref_list, sent_id):
    # references are special sentences.
    # print('in ref_list sentence')
    n_sentences = ""
    for ch in ref_list.children:
        ch_str = str(ch)
        if ch.name == 'ref':
            sub_text = ''
            for gch in ch.children:
                # if gch.name in ['citation-alternatives', 'element-citation', 'mixed-citation', 'nlm-citation',
                # 'note', 'x']:
                sub_text = sub_text + " " + " ".join([d.string for d in gch.descendants if d.string])
                # print('sub_text')
                # print(sub_text)
            sent_id, n_sentences_method = sentence_split(sub_text, sent_id)
            n_sentences = n_sentences + n_sentences_method
        elif ch.name in ["sec", "fig", "statement", "div", "boxed-text", "list", "list-item", "disp-quote", "speech",
                         "fn-group", "fn", "def-list", "def-item", "def", "ack", "array", "table-wrap", "table",
                         "tbody", "caption", "answer", "sec-meta", "glossary", "question", "question-wrap", "x"]:
            # print("in first elseif")
            sent_id, n_sentences_method = call_sentence_tags(ch, sent_id)
            n_sentences = n_sentences + ch_str[:ch_str.find('>') + 1] + n_sentences_method + "</" + gch.name + ">"
        else:
            n_sentences = n_sentences + ch_str
    return sent_id, n_sentences


def process_p_tag(gch, sent_id):
    n_sentences = ""
    p_children = gch.contents
    gch_str = str(gch)
    if len(p_children) == 1 and (not p_children[0].string) and (p_children[0].name in ["ext-link", "e-mail", "uri", "inline-supplementary-material",
                                           "related-article", "related-object", "address", "alternatives", "array",
                                           "funding-source", "inline-graphic"]):
        n_sentences = n_sentences + gch_str
    else:
        # print('P tag:\n')
        # print(p_tag)
        # print(p_tag[3:-4])
        # check if p contains graphic and boxed-text without text inside.
        sent_id, n_sentences_method = sentence_split(gch_str[gch_str.find('>')+1:-4], sent_id)
        # print(n_sentences_method)
        # print(sub_sent)
        n_sentences = n_sentences + gch_str[:gch_str.find('>') + 1] + n_sentences_method + "</" + gch.name + ">"
    return sent_id, n_sentences


def call_sentence_tags(ch, sent_id):
    #print('In call_sentence_tags')
    # print()
    n_sentences = ""
    for gch in ch.children:
        # print(gch.name)
        gch_str = str(gch)
        if gch.name in ['article-title', 'title', 'subtitle', 'trans-title', 'trans-subtitle', 'alt-title', 'label', 'td', 'th']:
            # assumption is that these tags only contain one sentence.
            # print("GCH")
            # print(str(gch))
            if gch.find('p', recursive=False):
                sent_id, n_sentences_method = call_sentence_tags(gch, sent_id)
                n_sentences = n_sentences + gch_str[:gch_str.find('>') + 1] + n_sentences_method + "</" + gch.name + ">"
            else:
                sub_text = [str(d) for d in gch.children]
                # print(sub_text)
                n_sentences = n_sentences + gch_str[:gch_str.find('>') + 1] + '<SENT id="' + str(
                    sent_id) + '"><plain>' + "".join(sub_text) + "</plain></SENT></" + gch.name + ">"
                sent_id = sent_id + 1
        elif gch.name in ["sec", "fig", "statement", "div", "boxed-text", "list", "list-item", "disp-quote", "speech",
                          "fn-group", "fn", "def-list", "def-item", "def", "ack", "array", "table-wrap", "table",
                          "tbody", "thead", "tr", "caption", "answer", "sec-meta", "glossary", "question", "question-wrap"]:
            # print("in first elseif : sec etc string")
            # print(gch_str)
            sent_id, n_sentences_method = call_sentence_tags(gch, sent_id)
            n_sentences = n_sentences + gch_str[:gch_str.find('>') + 1] + n_sentences_method + "</" + gch.name + ">"
            # print(gch_str)
            # print(gch_str[:gch_str.find('>')+1]) # works, tag.name will
            # print("n sentences method")
            # print(n_sentences_method)
            # print(gch_str[:gch_str.find('>')+1] + "".join(n_sentences_method) + "</" + gch.name + ">")
            # may need to do further changes
        elif gch.name == 'p':
            sent_id, n_sentences_method = process_p_tag(gch, sent_id)
            n_sentences = n_sentences + n_sentences_method
        else:
            # convert the xml into string and concat
            n_sentences = n_sentences + gch_str

    return sent_id, n_sentences


def process_front(front):
    #print('In process_front')
    sent_id = 1
    n_sentences = ""
    if front.find('article-meta'):
        # print('In article-meta')
        art_meta = front.find('article-meta')
        # print(art_meta)
        # directly using 0th item because according to the JATS DTD it should appears precisely once.

        for ch in art_meta.find_all(recursive=False):
            ch_str = str(ch)
            if ch.name in ['title-group', 'supplement', 'supplementary-material', 'abstract', 'trans-abstract',
                           'kwd-group', 'funding-group']:
                sent_id, n_sentences_method = call_sentence_tags(ch, sent_id)
                n_sentences = n_sentences + ch_str[:ch_str.find('>') + 1] + n_sentences_method + "</" + ch.name + ">"
            else:
                n_sentences = n_sentences + ch_str
        #print('\n\nend of process_front')
    return sent_id, n_sentences


def process_body(body, sent_id):
    # sent_id = 0
    #print('In process_body')
    n_sentences = ""
    for ch in body.find_all(recursive=False):
        ch_str = str(ch)
        if ch.name == 'p':
            sent_id, n_sentences_method = process_p_tag(ch, sent_id)
            n_sentences = n_sentences + n_sentences_method
        elif ch.name in ['sec', 'ack', 'alternatives', 'array', 'preformat', 'fig', 'fig-group' 'question-wrap',
             'question-wrap-group', 'list', 'table-wrap-group', 'table-wrap', 'display-formula',
             'display-formula-group', 'def-list', 'list', 'supplementary-material', 'kwd-group',
             'funding-group', 'statement', 'fig']:
            sent_id, n_sentences_method = call_sentence_tags(ch, sent_id)
            n_sentences = n_sentences + ch_str[:ch_str.find('>') + 1] + n_sentences_method + "</" + ch.name + ">"
        else:
            n_sentences = n_sentences + ch_str
    #print('\n\nend of process_body')
    return sent_id, n_sentences


def process_back(back, sent_id):
    # sent_id = 0
    #print('In process_back')
    n_sentences = ""
    for ch in back.find_all(recursive=False):
        ch_str = str(ch)
        if ch.name in ['sec', 'p', 'ack', 'alternatives', 'array', 'preformat', 'fig', 'fig-group' 'question-wrap',
             'question-wrap-group', 'list', 'table-wrap-group', 'table-wrap', 'display-formula',
             'display-formula-group', 'def-list', 'list', 'supplementary-material', 'kwd-group',
             'funding-group', 'statement', 'ref-list', 'glossary']:
            # print(ch.name)
            if ch.name == 'ref-list':
                #print('\n\n\n in ref-list')
                sent_id, n_sentences_method = reference_sents(ch, sent_id)
            else:
                sent_id, n_sentences_method = call_sentence_tags(ch, sent_id)
            n_sentences = n_sentences + ch_str[:ch_str.find('>') + 1] + n_sentences_method + "</" + ch.name + ">"
        else:
            n_sentences = n_sentences + ch_str
    #print('\n\nend of process_body')
    return sent_id, n_sentences


def process_full_text(each_file, out_file):  
    # replacing body tag with orig_body to stop BeautifulSoup from removing this tag.
    each_file = each_file.replace('<body>', '<orig_body>')
    each_file = each_file.replace('<body ', '<orig_body ')
    each_file = each_file.replace('</body>', '</orig_body>')
    try:
        xml_soup = BeautifulSoup(each_file, 'lxml')
        # BeautifulSoup adds an extra html and body tag. removing those 2 tags.
        if xml_soup.html:
            # print('html unwrap')
            xml_soup.html.unwrap()
        if xml_soup.body:
            # print('body unwrap')
            xml_soup.body.unwrap()
        # print(xml_soup.find('front'))
        if xml_soup.find('orig_body'):
            # print('orig_body find')
            xml_soup.find('orig_body').name = 'body'
        sent_id = 1
        xml_string = None
        # Access regions of interests for sentence splitting.
        last_tag=None
        if xml_soup.article.find('front'):
            index = str(xml_soup).index('<front>')
            # print("\n\nFront index: " + str(index))
            xml_string = str(xml_soup)[:index] + "<front>"
            # print("Pre-front")
            # print(xml_string)
            sent_id, n_sentences_front = process_front(xml_soup.article.find('front'))
            xml_string = xml_string + n_sentences_front + "</front>"
            last_tag = "</front>"
        if xml_soup.article.find('body'):
            if xml_string == None:
                index = str(xml_soup).index('<body>')
                # print("\n\nBody index: " + str(index))
                xml_string = str(xml_soup)[:index]
                # print("Pre-body")
                # print(xml_string)
            sent_id, n_sentences_body = process_body(xml_soup.article.find('body'), sent_id)
            xml_string = xml_string + "<body>" + n_sentences_body + "</body>"
            last_tag = "</body>"
        if xml_soup.article.find('back'):
            if xml_string == None:
                index = str(xml_soup).index('<back>')
                # print("\n\nBack index: " + str(index))
                xml_string = str(xml_soup)[:index]
                # print("Pre-back")
                # print(xml_string)
            sent_id, n_sentences_back = process_back(xml_soup.article.find('back'), sent_id)
            xml_string = xml_string + "<back>" + n_sentences_back + "</back>"
            '''    
                print('\n\n\nFront sentences:\n')
                print(n_sentences_front)
                print('\n\n\nBody sentences:\n')
                print(n_sentences_body)
                print('\n\n\nBack sentences:\n')
                print(n_sentences_back)
            '''
            last_tag = "</back>"
        if last_tag:
            index_last_tag = str(xml_soup).index(last_tag)
            xml_string = xml_string + str(xml_soup)[index_last_tag + len(last_tag):]
        else:
            print("No FRONT, BODY BACK.")
            xml_string = str(xml_soup)
        with open(out_file, 'a') as out:
            out.write(str(xml_string) + "\n")
            # break

    except Exception as e:
        print(e)

def process_abstract_text(gch, sent_id, gch_flag):
    n_sentences = ""
    p_children = gch.contents
    gch_str = str(gch)
    sent_id, n_sentences_method = sentence_split(gch_str[gch_str.find('>') + 1:-(len("</" + gch.name + ">"))], sent_id)
    #print("n sentences in process anbstract text")
    #print(n_sentences_method)
    # print(sub_sent)
    if gch_flag==1:
        n_sentences = n_sentences + gch_str[:gch_str.find('>') + 1] + n_sentences_method + "</" + gch.name + ">"
    else:
        n_sentences = n_sentences + n_sentences_method
    return sent_id, n_sentences


def process_abstracts(each_file, out_file):
    sent_id = 1
    try:
        xml_soup = BeautifulSoup(each_file, 'lxml')
        # BeautifulSoup adds an extra html and body tag. removing those 2 tags.
        if xml_soup.html:
            # print('html unwrap')
            xml_soup.html.unwrap()
        if xml_soup.body:
            # print('body unwrap')
            xml_soup.body.unwrap()
        n_sentences = ""
        #print(str(xml_soup))
        target_tags = xml_soup.article.find_all(["abstract"])
        if len(target_tags)>0:
            # abstract_tags = xml_soup.pubmedarticle.medlinecitation.find_all(["abstract", "otherabstract"], recursive=False)
            prev = None
            xml_str = str(xml_soup)
            #print(target_tags)
            if target_tags:
                # print("targets found")
                # print(abstract_tags)
                ch_tag_end = 0
                prev_start_index=0
                prev=None
                for ch in target_tags:
                    #print(ch.name)
                    ch_str = str(ch)
                    match = re.search("<" + ch.name +"[^>]*?>", xml_str)
                    if match:
                        #print(match.group())
                        ch_tag_end = match.end()
                        #print("ch_tag_end:" + str(ch_tag_end))
                    if prev==None:
                        n_sentences = xml_str[:ch_tag_end]
                        prev = ch
                    else:
                        # find end of previous ch
                        #print("previous name:" + prev.name)
                        match = re.search("</"+prev.name+">", xml_str)
                        if match:
                            prev_start_index = match.start()
                        # print("prev_start:" + str(prev_start_index))
                        # print(match)
                        n_sentences = n_sentences + xml_str[prev_start_index:ch_tag_end]
                        prev=ch
                    # print("before gchildren loop")
                    # print(n_sentences)
                    # print("\nEND of n_sentences")
                    if ch.name == "title":
                        #print("article title")
                        sent_id, n_sentences_method = process_abstract_text(ch, sent_id, 0)
                        #print(n_sentences_method)
                        n_sentences = n_sentences + n_sentences_method
                    else:
                        for gch in ch.find_all(recursive=False):
                            gch_str = str(gch)
                            if gch.name=='title' or gch.name=='p':
                                #print("abstracttext")
                                sent_id, n_sentences_method = process_abstract_text(gch, sent_id, 1)
                                #print(n_sentences_method)
                                n_sentences = n_sentences + n_sentences_method
                            else:
                                n_sentences = n_sentences + gch_str
                if prev:
                    match = re.search("</"+prev.name+">", xml_str)
                    if match:
                        prev_start_index = match.start()
                    n_sentences = n_sentences + xml_str[prev_start_index:]
                else:
                    print("\nERROR: Prev should not be none")
            else:
                print("No Abstract or Other Abstract\n")
                n_sentences=xml_str
        with open(out_file, 'a') as out:
            out.write(n_sentences)
    except Exception as e:
        print(e)
    

def process_each_article(each_file_path, out_file, document_flag):
    files_list = getfileblocks(each_file_path, document_flag)
    with open(out_file, 'w') as out:
        print(out_file)
        if document_flag=='a':
            out.write("<articles>\n")
    for each_file in tqdm(files_list):
        if document_flag=='f':
            process_full_text(each_file, out_file)
        elif document_flag=='a':
            print("abstract")
            process_abstracts(each_file, out_file)
        else:
            print("Wrong document Flag")
    if document_flag=='a':
        with open(out_file, 'a') as out:
            out.write("</articles>")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script will process XML articles in JAT-Archive DTD and do sentence splitting')
    parser.add_argument("-f", "--file", required=True, help="XML file", metavar="PATH")
    parser.add_argument("-o", "--out", required=True, help="Out file", metavar="PATH")
    parser.add_argument("-d", "--document", required=True, help="Out file", metavar="PATH")
    args = parser.parse_args()
    print(args.file)
    process_each_article(args.file, args.out, args.document)
    print(args.file + ": Sentenciser Finished!")
