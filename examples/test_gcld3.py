import time

import gcld3

detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                        max_num_bytes=1000)
# text = "This text is written in English"
text = "薄雾"
while True:
    result = detector.FindLanguage(text=text)
    print(text, result.probability, result.language)
    time.sleep(0.01)
