import pandas as pd
import json
import argparse
from simpletransformers.classification import ClassificationModel, ClassificationArgs


def association_classification(json_file, out_json, model):
    entity_map = {'GP': 'OTAR_TARGET', 'DS': 'OTAR_DISEASE', 'OG': 'OTAR_ORGANISM'}
    co_occur_flag = 0
    annot_map = {0: 'YGD', 1: 'NGD'}
    with open(json_file) as json_fh, open(out_json, 'w') as write_json:
        while True:
            line = json_fh.readline()
            if not line:
                break
            article_annot = json.loads(line)
            article_annot['pprid']=""
            for sent in article_annot['sentences']:
                if 'co-occurrence' in sent:
                    sent_text = sent['text']
                    co_occur_flag = 1
                    # print("Original Sentence:" + sent_text)
                    for association in sent['co-occurrence']:
                        if association['type'] == 'GP-DS':
                            #print("Original Sentence:" + sent_text)
                            if association['start1'] < association['start2']:
                                sent_temp = sent_text[0:association['start1']] + 'OTAR_TARGET' + sent_text[association['end1']:association['start2']] + 'OTAR_DISEASE' + sent_text[association['end2']:]
                                #print('After Replace:' + sent_temp)
                                predictions, raw_outputs = model.predict([sent_temp])
                                association['association'] = annot_map[predictions[0]]
                                #print(predictions)
                            else:
                                sent_temp = sent_text[0:association['start1']] + 'OTAR_DISEASE' + sent_text[association['end1']:association['start2']] + 'OTAR_TARGET' + sent_text[association['end2']:]
                                #print('After Replace:' + sent_temp)
                                predictions, raw_outputs = model.predict([sent_temp])
                                association['association'] = annot_map[predictions[0]]
                                #print(predictions)

            json.dump(article_annot, write_json)
            write_json.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script will process Gold Standard in JSON format and extract the association sentences and save in TSV format.')
    parser.add_argument("-f", "--file", nargs=1, required=True, help="Gold Standard in JSON format", metavar="PATH")
    parser.add_argument("-o", "--out", nargs=1, required=True, help="Output file", metavar="PATH")
    parser.add_argument("-m", "--model", nargs=1, required=True, help="Model PATH", metavar="PATH")

    args = parser.parse_args()
    # print(args.out[0])
    # model_path = '/nfs/production/literature/literature_otar/shyama/Association/model/checkpoint-680-epoch-5/'
    model_args = ClassificationArgs(num_train_epochs=5, overwrite_output_dir=True)
    model = ClassificationModel(
    'bert',
    args.model[0],
    num_labels=2,
    use_cuda=False,
    args=model_args
    )
    association_classification(args.file[0], args.out[0], model)
    print(args.file[0] + ' : GP-DS association classification finished!')
