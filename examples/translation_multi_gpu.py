"""
This script show how we can do translation using multiple processes

Usage:
python translation_speed.py model_name
"""
import csv
import gzip
import logging
import os
import sys
import time

from easynmt import util, EasyNMT

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    model = EasyNMT(sys.argv[1])

    nli_dataset_path = 'AllNLI.tsv.gz'
    sentences = set()

    snli_max_sentences = 2000
    mnli_max_sentences = 2000
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

    ######## Multi-Process-Translation
    # You can pass a target_devices parameter to the start_multi_process_pool() method to define how many processes to start
    # and on which devices the processes should run
    process_pool = model.start_multi_process_pool(['cuda', 'cuda'])

    # Do some warm-up
    model.translate_multi_process(process_pool, sentences[0:100], source_lang='en', target_lang='de',
                                  show_progress_bar=False)

    # Start translation speed measure - Multi process
    start_time = time.time()
    translations_multi_p = model.translate_multi_process(process_pool, sentences, source_lang='en', target_lang='de',
                                                         show_progress_bar=True)
    end_time = time.time()
    print("Multi-Process translation done after {:.2f} sec. {:.2f} sentences / second".format(end_time - start_time,
                                                                                              len(sentences) / (
                                                                                                          end_time - start_time)))

    model.stop_multi_process_pool(process_pool)
