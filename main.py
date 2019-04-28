import string
import random
import os
import sys

import nltk
import dill
from Classifier import ClassifierBasedGermanTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from germalemma import GermaLemma

# Pickle file name
TAGGER_PKL = 'tagger.pkl'
CORPUS_PKL = 'corpus.pkl'


def remove_punctuation(words):
    punct = set(string.ascii_letters + string.whitespace + "üäößÜÄÖ")
    chars_with_spaces = "".join([char for char in words if char in punct])
    return " ".join(chars_with_spaces.split())


def load_words(filename, remove_punct=False):
    with open(filename, 'r', encoding="utf-8") as riddle:
        words = riddle.read()

        if remove_punct:
            # remove everything that isn't a space or a-zA-Z
            words = remove_punctuation(words)

        words = words.split()

    return words


def remove_stopwords(words, stopwords):
    words = [word for word in words if word.lower() not in stopwords]
    return words


def make_tagger():
    if os.path.exists(TAGGER_PKL):
        tagger = dill.load(open(TAGGER_PKL, 'rb'))
    else:
        # obtain the training corpus "TIGER Corpus"
        # https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html
        corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09',
                                             ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                             encoding='utf-8')

        # divides up the tagged words into sentences
        tagged_sents = list(corp.tagged_sents())
        random.shuffle(tagged_sents)

        # split: 10% for testing and 90 % for training
        split_perc = 0.1
        split_size = int(len(tagged_sents) * split_perc)
        train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]

        # use a tagger to train
        tagger = ClassifierBasedGermanTagger(train=train_sents)

        # save the tagger (pickle) in binary "wb"
        dill.dump(tagger, open(TAGGER_PKL, 'wb'))

    return tagger


def lemmatize(lemmatizer, tagged_words):
    new_words = []
    for (word, tag) in tagged_words:
        try:
            new_words += [(lemmatizer.find_lemma(word, tag))]
        except ValueError:
            # we still want the words that could not be lemmatized
            new_words += [word]

    return new_words


def tag_and_lemma(words, tagger=None, lemmatizer=None):

    # pos tagging
    words = tagger.tag(words)

    # lemmatize the words, e.g. Briefen => Brief
    words = lemmatize(lemmatizer, words)

    return words


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def get_top_n_words(corpus, n=None):
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                       reverse=True)
    return words_freq[:n]


def main():

    # remove stop words
    words = load_words('four.txt', remove_punct=True)
    total_words = len(words)
    print('Found %d words' % total_words)
    stopwords = set(load_words("stopwords_de_full.txt", remove_punct=False))

    words = remove_stopwords(words, stopwords)
    print('Removed %d stop words' % (total_words - len(words)))

    tagger = make_tagger()
    lemmatizer = GermaLemma()

    words = tag_and_lemma(words, tagger=tagger, lemmatizer=lemmatizer)

    if os.path.exists(CORPUS_PKL):
        lemmatized_sents = dill.load(open(CORPUS_PKL, 'rb'))
    else:
    # read each line in a list
        with open('deu_news_2014_30K-sentences.txt',  'r', encoding="utf-8") as f:
            sentences = f.readlines()
            sentences = [x.strip() for x in sentences]
            lemmatized_sents = []

            for i, sentence in enumerate(sentences):
                sentence = remove_punctuation(sentence)
                sentence_words = tag_and_lemma(sentence.split(), tagger=tagger, lemmatizer=lemmatizer)
                lemmatized_sents += [sentence_words]

                if i % 100 == 0:
                    print(".", end='')
                    sys.stdout.flush()

            print("")

        dill.dump(lemmatized_sents, open(CORPUS_PKL, 'wb'))

    print("Using " + str(len(lemmatized_sents)) + " sentences in the corpus")

    vectorizer = TfidfVectorizer(smooth_idf=True, sublinear_tf=True, norm=None, analyzer='word')

    lemmatized_sents = [remove_stopwords(sent, stopwords) for sent in lemmatized_sents]
    reconstructed = [" ".join(ls) for ls in lemmatized_sents] + words

    fit = vectorizer.fit(reconstructed)
    feature_names = vectorizer.get_feature_names()
    transform = fit.transform(words)
    sorted_items = sort_coo(transform.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    for k in keywords:
        print(k, keywords[k])


if __name__ == "__main__":
    main()


