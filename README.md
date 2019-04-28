# Riddle Solver
(This solver needs a  better German corpus (with bundled texts) to realize it. The current corpus used from http://wortschatz.uni-leipzig.de/en is only a sentence-based corpus and is insufficient to support the solver.)
### Process the riddle document
1. Remove the noise (e.g., punctuation)
2. Remove the stopwords (https://github.com/solariz/german_stopwords)
3. Pos (part-of-speech) tag with the Tiger Corpus (https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html)
4. Lemmatize the words (https://github.com/WZBSocialScienceCenter/germalemma)
### Caculate TFIDF to extract keywords from the riddle
1. Add a German text corpus (http://wortschatz.uni-leipzig.de/en)
2. Remove the noise and stopwords in the Corpus
3. Lemmatize the words in the Corpus
4. Combine the corpus with the riddle to generate a reconstructed corpus
5. Create a vectorizor by TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
6. Learn and build the vocabulary (features) from the reconstructed corpus
7. Generate the tf-idf for the riddle
8. Extract the top 10 keywords
