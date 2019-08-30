import re
from typing import List

from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import split_alphanum
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import strip_tags
import tools.model.model_classes as merm_model
import tools.utils.envutils as env
import tools.utils.log as log


def clean_string_for_tokenizing(textIn):
    cleaned = gensim_clean_string(textIn)
    cleaned = cleaned.replace("'s", "")
    cleaned = cleaned.replace("'", "")
    cleaned = cleaned.replace("’", "")
    cleaned = cleaned.replace("\\n", "")
    #cleaned = _cleaner_text(str(textIn))
    cleaned = re.sub("b'", ' ', cleaned)
    regex = re.compile(r'[^a-zA-Z0-9 ]')
    cleaned = regex.sub(" ", cleaned)


    #cleaned = re.sub(' +', ' ', cleaned)
    cleaned = re.sub('\.+', '.', cleaned)




    return cleaned.lower().strip()


def gensim_clean_string(textIn):
    cleaner = strip_tags(textIn)
    cleaner = split_alphanum(cleaner)
    cleaner = strip_multiple_whitespaces(cleaner)
    cleaner = strip_short(cleaner,minsize=3)
    return cleaner

def clean_raw_content(textIn):
    cleaner = textIn.replace("\\n", "")
    cleaner = strip_tags(cleaner)
    cleaner = strip_multiple_whitespaces(cleaner)
    return cleaner



def cleanstring_simple(s):
    clean = s.replace('"', '').replace("\\n", "").replace(":", ";").replace("\\", " ")
    return clean


def lemmatize_tokens(corpora_list: List[merm_model.LinkedDocument], stop_words: List[str]):

    stoplist = stop_words
    lemmatized_corpus = []
    iter_count = 0
    lemmatizer = WordNetLemmatizer()
    #log.getLogger().info("Lemmatizing corpus. This can be slow.")
    for doc in corpora_list:
        lemmatized_text = []
        for word in doc.tokens:
            # print("word: " + word)
            lemmatized_word = lemmatizer.lemmatize(word)
            if lemmatized_word is not None:
                cleanword = clean_string_for_tokenizing(lemmatized_word)
                if cleanword not in stoplist and len(cleanword) > 1 and not hasNumbers(cleanword):
                    # print(cleanword)
                    lemmatized_text.append(cleanword)
        doc.tokens = lemmatized_text
        lemmatized_corpus.append(doc)
        iter_count += 1
        #sys.stdout.write(".")
        if env.test_env() == True and iter_count > env.test_env_doc_processing_count():
            log.getLogger().info("DEV MODE: Breaking loop here")
            break
    #sys.stdout.flush()
    return lemmatized_corpus

def tokenize(corpora_list:List[merm_model.LinkedDocument]):
    for linked_doc in corpora_list:

        linked_doc.tokens = clean_string_for_tokenizing(linked_doc.raw.lower()).split()

    return corpora_list

def split_linked_doc_by_sentence(linked_doc: merm_model.LinkedDocument):
    linked_sentence_list = []
    raw_sentences = linked_doc.raw.split(".")
    for sentence in raw_sentences:
        linked_sentence = merm_model.LinkedDocument(sentence, linked_doc.title, [], linked_doc.source, linked_doc.ui,
                                                      linked_doc.provider,  linked_doc.uid, linked_doc.index_name,
                                                      linked_doc.space, linked_doc.scores, linked_doc.corpus_doc,
                                                      linked_doc.any_analysis,linked_doc.updated)
        linked_sentence_list.append(linked_sentence)
    return linked_sentence_list


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def standard_stop_words():
    return ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
            "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
            "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a",
            "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
            "about",
            "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
            "up",
            "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
            "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
            "don",
            "should", "now", "n", "html", "p", "div", "li", "val", "def", "id", "quot", "http", "com", "merm", "data", "file",
            "skin", "tone", "slightly", "smiling", "face", "thumbs", "open", "rescoped", "opened", "commented", "updated", "rescope",
            "open", "comment", "update"]


