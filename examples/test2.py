# PYTHONPATH=$PWD python examples/test2.py m2m_100_418M
# PYTHONPATH=$PWD python examples/test2.py m2m_100_1.2B
# PYTHONPATH=$PWD python examples/test2.py opus-mt
import sys
import time

from easynmt import EasyNMT

a = 'An older man dressed in blue historical clothing is ringing a bell in his right hand.'
model = EasyNMT(sys.argv[1])

sentences = []
for i in range(100):
    sentences.append(a)

start_time = time.time()
model.translate(sentences, source_lang='en', target_lang='de', show_progress_bar=True,
                perform_sentence_splitting=False)
end_time = time.time()
print("Done after {:.2f} sec. {:.2f} sentences / second".format(end_time - start_time,
                                                                len(sentences) / (end_time - start_time)))
