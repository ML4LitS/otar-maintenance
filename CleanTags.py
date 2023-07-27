import argparse
import io
from bs4 import BeautifulSoup
from tqdm import tqdm


# read xml files
def getfileblocks(file_path, document_flag):
    sub_file_blocks = []
    if document_flag == 'f':
        xml = '<?xml version="1.0" encoding="UTF-8"?>'
        start_str = '<article '
        start_str2 = '<articles><article '
        try:
            with io.open(file_path, 'r', encoding='utf8') as fh:
                for line in fh:
                    if line.startswith(start_str) or line.startswith(start_str2):
                        # sub_file_blocks.append(xml)
                        sub_file_blocks.append(line.replace('^<articles>', ''))
                    elif len(line.strip())>0:
                        sub_file_blocks[-1] += line.strip().replace('</articles>$', '')
                    #else:
                    #    print("Empty line:" + line)
        except Exception as e:
            #print('error processing, skipping file : ' + str(len(sub_file_blocks)))
            with io.open(file_path, 'r', encoding='ISO-8859-1') as fh:
                for line in fh:
                    if line.startswith(start_str) or line.startswith(start_str2):
                        sub_file_blocks.append(line.replace('^<articles>', ''))
                    elif len(line.strip())>0:
                        sub_file_blocks[-1] += line.strip().replace('</articles>$', '')
                    #else:
                    #    print("Empty line:" + line)
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

    return sub_file_blocks


def make_plain(soup):
    for plain in soup.find_all('plain'):
        for ch in plain.find_all():
            ch.unwrap()
    return soup


def process_each_article(each_file_path, out_file, document_flag):
    files_list = getfileblocks(each_file_path, document_flag)
    with open(out_file, 'w') as out:
        #print(out_file)
        if document_flag=='a':
            out.write("<articles>\n")
    for each_file in tqdm(files_list):
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
            # check if the soup is valid
            soup = make_plain(xml_soup)
            with open(out_file, 'a') as out:
                out.write(str(soup) + "\n")
            # print(soup)
        except Exception as e:
            print(e)
    if document_flag=='a':
        with open(out_file, 'a') as out:
            out.write("</articles>")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script will process sentencised XML articles in JATs DTD and remove any tags from the SENT tag')
    parser.add_argument("-f", "--file", required=True, help="XML file", metavar="PATH")
    parser.add_argument("-o", "--out", required=True, help="Out file", metavar="PATH")
    parser.add_argument("-d", "--document", required=True, help="Document type (f|a)", metavar="PATH")
    args = parser.parse_args()
    #print(args.file)
    process_each_article(args.file, args.out, args.document)
    print(args.file + ": Tag cleaning finished")
