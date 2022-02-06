import datetime

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = 'facebook/m2m100_418M'
# model_name = 'facebook/m2m100_1.2B'
# 100种语言包括: Afrikaans (af), Amharic (am), Arabic (ar), Asturian (ast), Azerbaijani (az), Bashkir (ba), Belarusian (be), Bulgarian (bg), Bengali (bn), Breton (br), Bosnian (bs), Catalan; Valencian (ca), Cebuano (ceb), Czech (cs), Welsh (cy), Danish (da), German (de), Greeek (el), English (en), Spanish (es), Estonian (et), Persian (fa), Fulah (ff), Finnish (fi), French (fr), Western Frisian (fy), Irish (ga), Gaelic; Scottish Gaelic (gd), Galician (gl), Gujarati (gu), Hausa (ha), Hebrew (he), Hindi (hi), Croatian (hr), Haitian; Haitian Creole (ht), Hungarian (hu), Armenian (hy), Indonesian (id), Igbo (ig), Iloko (ilo), Icelandic (is), Italian (it), Japanese (ja), Javanese (jv), Georgian (ka), Kazakh (kk), Central Khmer (km), Kannada (kn), Korean (ko), Luxembourgish; Letzeburgesch (lb), Ganda (lg), Lingala (ln), Lao (lo), Lithuanian (lt), Latvian (lv), Malagasy (mg), Macedonian (mk), Malayalam (ml), Mongolian (mn), Marathi (mr), Malay (ms), Burmese (my), Nepali (ne), Dutch; Flemish (nl), Norwegian (no), Northern Sotho (ns), Occitan (post 1500) (oc), Oriya (or), Panjabi; Punjabi (pa), Polish (pl), Pushto; Pashto (ps), Portuguese (pt), Romanian; Moldavian; Moldovan (ro), Russian (ru), Sindhi (sd), Sinhala; Sinhalese (si), Slovak (sk), Slovenian (sl), Somali (so), Albanian (sq), Serbian (sr), Swati (ss), Sundanese (su), Swedish (sv), Swahili (sw), Tamil (ta), Thai (th), Tagalog (tl), Tswana (tn), Turkish (tr), Ukrainian (uk), Urdu (ur), Uzbek (uz), Vietnamese (vi), Wolof (wo), Xhosa (xh), Yiddish (yi), Yoruba (yo), Chinese (zh), Zulu (zu)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
a = 'An older man dressed in blue historical clothing is ringing a bell in his right hand.'
article_cn = "生活就像一块巧克力。"
en_lang_id = tokenizer.get_lang_id("en")

article_en = "Life is like a box of chocolate."
zh_lang_id = tokenizer.get_lang_id("zh")


def cn2en():
    tokenizer.src_lang = "zh"
    encoded_text = tokenizer(article_cn, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=en_lang_id)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # print(f"{article_cn}的翻译结果: {result[0]}")


def en2cn():
    tokenizer.src_lang = "de"
    encoded_text = tokenizer(a, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=zh_lang_id)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # print(f"{article_en}的翻译结果: {result[0]}")


# cn2en()
print(datetime.datetime.now())
for i in range(1000):
    en2cn()
    # cn2en()
print(datetime.datetime.now())
