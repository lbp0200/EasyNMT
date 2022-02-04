"""
This script measures the translation speed.

Usage:
python translation_speed.py model_name
"""
import csv
import gzip
import os
import sys
import time

from easynmt import util, EasyNMT

model = EasyNMT(sys.argv[1])

nli_dataset_path = 'AllNLI.tsv.gz'
sentences = set()

snli_max_sentences = 1000
mnli_max_sentences = 1000
snli = 0
mnli = 0

# Download datasets if needed
if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['sentence1'] in sentences or len(row['sentence1']) > 200:
            continue

        if len(model.sentence_splitting(row['sentence1'])) > 1:
            continue

        if row['dataset'] == 'SNLI' and snli < snli_max_sentences:
            sentences.add(row['sentence1'])
            snli += 1
        if row['dataset'] == 'MNLI' and mnli < mnli_max_sentences:
            sentences.add(row['sentence1'])
            mnli += 1
        if snli >= snli_max_sentences and mnli >= mnli_max_sentences:
            break

print("Sentences:", len(sentences))
sentences = list(sentences)

# Some warm up
# start_time = time.time()
# model.translate(sentences[0:100], source_lang='en', target_lang='de', perform_sentence_splitting=False)
# end_time = time.time()
# print("Done after {:.2f} sec. {:.2f} sentences / second".format(end_time - start_time, 100 / (end_time - start_time)))

# Start translation speed measure
start_time = time.time()
model.translate(sentences, source_lang='en', target_lang='de', show_progress_bar=True,
                perform_sentence_splitting=False)
end_time = time.time()
print("Done after {:.2f} sec. {:.2f} sentences / second".format(end_time - start_time,
                                                                len(sentences) / (end_time - start_time)))
