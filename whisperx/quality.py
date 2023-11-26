import re
import zlib

IDEOGRAPHIC_SPACE = 0x3000
NUM_WORDS_PER_SECOND = 2.0

def is_asian(char):
    """Is the character Asian?"""
    return ord(char) > IDEOGRAPHIC_SPACE

def filter_jchars(c):
    """Filters Asian characters to spaces"""
    if is_asian(c):
        return ' '
    return c

def nonj_len(word):
    u"""Returns number of non-Asian words in {word}
    – 日本語AアジアンB -> 2
    – hello -> 1
    @param word: A word, possibly containing Asian characters
    """
    # Here are the steps:
    # 日spam本eggs
    # -> [' ', 's', 'p', 'a', 'm', ' ', 'e', 'g', 'g', 's']
    # -> ' spam eggs'
    # -> ['spam', 'eggs']
    # The length of which is 2!
    chars = [filter_jchars(c) for c in word]
    return len(''.join(chars).split())

def get_wordcount(text):
    """Get the word/character count for text

    @param text: The text of the segment
    """

    characters = len(text)
    chars_no_spaces = sum([not x.isspace() for x in text])
    asian_chars =  sum([is_asian(x) for x in text])
    non_asian_words = nonj_len(text)
    words = non_asian_words + asian_chars

    return dict(characters=characters,
                chars_no_spaces=chars_no_spaces,
                asian_chars=asian_chars,
                non_asian_words=non_asian_words,
                words=words)

def dict2obj(dictionary):
    """Transform a dictionary into an object"""
    class Obj(object):
        def __init__(self, dictionary):
            self.__dict__.update(dictionary)
    return Obj(dictionary)

def get_wordcount_obj(text):
    """Get the wordcount as an object rather than a dictionary"""
    return dict2obj(get_wordcount(text))

def no_punct(text, word_count) -> bool:
    matches = re.findall("[,.?!，。？！]", text)
    return len(matches) == 0 or word_count / len(matches) >= 40

def language_mismatch(seg, lang: str|None) -> bool:
    if not lang:
        return False
    seg_lang = seg["language"]
    if isinstance(seg_lang, str):
        return seg_lang != lang
    else:
        return seg_lang[0] != lang

def low_word_density(seg) -> bool:
    active_duration = seg["active_duration"]
    seg_duration = seg["end"] - seg["start"]
    if seg_duration >= 3 and active_duration < seg_duration * 0.8:
        return True
    return seg["word_count"] < seg_duration * NUM_WORDS_PER_SECOND

def high_compression_ratio(text) -> bool:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes)) > 2.4

def maybe_reject_reason(seg, lang=None) -> str:
    text = seg["text"].strip()
    num_words = get_wordcount_obj(text).words
    seg["word_count"] = num_words
    # Ranked by how critical the problem is.
    if seg["logprob"] <= -1.0:
        reason = "low_logprob"
    elif seg["no_speech_prob"] >= 0.9:
        reason = "no_speech"
    elif language_mismatch(seg, lang):
        reason = "language_mismatch"
    elif low_word_density(seg):
        reason = "low_word_density"
    elif high_compression_ratio(text):
        reason = "high_compression_ratio"
    elif no_punct(text, num_words):
        reason = "no_punctuations"
    else:
        reason = ""
    return reason
