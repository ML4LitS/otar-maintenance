# OpenTargets
As part of the Open Targets literature mining project, Europe PMC team currently annotates Medline abtracts, pre-prints abstract from a number of sources and full-text articles from PubMed Central with BioMedical Entities and provides Entity association for the Open Targets platform. Europe PMC team cutrrently mines Gene/Proteins, Diseases, Chemicals and Organisms using BioBERT model. Evidence sentence/string is defined by the co-occurence of different Entity types in a sentence. For example, "Telethonin/T-CAP has taken on added importance due to its causal role in limb-girdle muscular dystrophy type 2G (Moreira et al., 2000)." is an evidence sentence describing association between Target (Gene/Proteins) "T-CAP" and Disease "limb-girdle muscular dystrophy type 2G". This repository describes the core Named Entity Recognition (NER) and Target-Disease association classification.
<br>
### Sentence boundary

The first component of the Open Targets Association identification is the Sentenciser. We need to identify the sentence boundaries as our entity association is sentence based. The Sentenciser (Sentenciser.py) takes GZip XML file containing multiple abstracts and full-text articles as input and produces XML file with sentence boundaries marked by SENT tags. Abstracts and the Full-text have different file formats, please check the examples folder of the repo for the examples. This scripts take three command-line arguments, 
- -f : Input file path [GZIP XML file] 
- -o : Full path for the output XML file
- -d : The input file type i.e. whether its abstract or full-text. Please use 'a' for the abstracts and 'f' for the full-text.   

### Cleaning formatting and other tags

The next step in the pipeline is to remove any XML tags appearing with the SENT tag and improve accuracy of NER. This script (CleanTags.py) takes the output of the Sentenciser as input using the -f argument and saves the output file mentioned after the -o option.

### Identifying Sections form the Full-text articles

We run a section tagger (OTAR_new_pipeline_section_tagger.py) for the full-text to identify the location of the evidence string within an article, e.g. Introduction, Result, Conclusion etc. This help the Open Targets platform to rank the articles based on the importance of evidence string. This script again takes two command line argumnets

- -f : Input XML file path which is output of the CleanTags.py script
- -o : Output XML file path

### NER identification

Europe PMC team developed a developed a gold standard dataset of 300 full-text articles for three BioMedical Entities, namely Gene/Proteins, Diseases, and Organisms. BioBERT language model was fine tuned for the NER task using this dataset. Later, using publicly available dataset Chemicals, a NER model was generated and used to annotate the 300 full-text articles with Chemicals. Europe PMC team used this annotation to generate the model to annotate all four Entities for the Open Targets platform and identify association between Targets (Gene/Proteins), Diseases and Chemicals. Our NER tagger (OTAR_new_pipeline_cluster_all.py) currently annotates the Entities mentioned earlier and finds the evidence strings to report them in JSON format. The output file is a multiline JSON file where each line represents a JSON object for an article. This script takes five command-line arguments,

- -f : the full path for the input file [i.e. the output of the section tagger (OTAR_new_pipeline_section_tagger.py) for the full-text or the output of the CleanTags.py for the abstracts]
- -o : Output Directory
- -m : NER model directory, e.g. 1604049631/ for current repo.
- -l : TSV file path containing mapping between PMIDs and PMCIDs and their publication dates. Please refer to the examples folder for an example of this file.
- -d : Possibles values are [a|f], please use 'a' for abstracts and 'f' for full-text.

### Target-Disease association classification

The component of this pipeline classify Target-Disease evidence strings using a BioBERT model. The manual curators annotated about 1K evidence strings from the gold-standard dataset. The annotators tagged the evidence strings using three tags.

- YGD : When the evidence string describe an association between the target and the disease mentioned in the sentence. Association can be positive or negetive.
- NGD : When the evidence string does not describe an association between the target and the disease mentioned in the sentence.
- AMB : When there is not enough information available in the sentence to identify the association.
We fine-tuned BioBERT model using SimpleTransformer package and classify only the Target-Disease association sentences as part of this pipeline due to the available dataset. Our association annotation script (otar_association_annotation.py) takes three arguments.

- -f : Input JSON file path, i.e. the output of the NER code [OTAR_new_pipeline_cluster_all.py].
- -o : Output JSON file path.
- -m : Association model directory, for this repo 'association_model' folder.

## Training association model
Please checkout the association folder for the code used to train and test the association classifier.

<br>
Please refer to https://gitlab.ebi.ac.uk/saha/opentargetsorchestrator see how Europe PMC team run this pipeline daily in an automated manner.

## Abstract Pipeline
<br>
<img src="image/abstract.png" width="300" height="500">
<br>

## Full-text Pipeline
<img src="image/full-text.png" width="300" height="500">

