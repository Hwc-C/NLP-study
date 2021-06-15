# Chapter 1. Essentials of NLP

## **Typical text processing workflow**

Data collection -> Labeling -> Text Normalization -> Vectorization -> Modeling


## **Data collection and labeling**

The first step of ML project is to obtain a dataset. Using Library likes scrapy or Beautiful Soup to scrap data from the website. 

Then labeling, label the data with a label that acts as a ground truth. 

> SMS Spam Collection [link](http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
> 
> Datasets from UCI [link](http://archive.ics.uci.edu/ml/datasets.php)
> 
> Other datasets [link](https://github.com/niderhoff/nlp-datasets)

```python
# Download the zip file
path_to_zip = tf.keras.utils.get_file("smsspamcollection.zip", origin="https://archive.ics.uci.edu/ml/machine-learningdatabases/00228/smsspamcollection.zip",extract=True)

# Unzip the file into a folder
!unzip $path_to_zip -d data
```

## **Text normalization**

**Main steps**: 
- case normaization, uppercase/lowercase letters OR remove punctuation
- tokenization
- stop word removal
- Parts-of-Speech tagging
- stemming


## **Modeling normalized data**

Using tensorflow

## **Tokenization**

If the input is a sentence, then separating the words would be an example of tokenization. 

```
!pip install stanfordnlp

import stanfordnlp as snlp
en = snlp.download("en")
def word_counts(x, pipeline=en):
    doc = pipeline(x)
    count = sum([len(sentence.tokens) for sentence in doc.sentences])
    return count
```

## **Stop word removal

Removing common words such as articles (the, an) and conjunctions (and, but), among others. 

```
!pip install stopwordsiso

import stopwordsiso as stopwords
stopwords.langs()

en_sw = stopwords.stopwords("en")
def word_counts(x, pipeline=en):
    doc = pipeline(x)
    count = 0
    for sentence in doc.sentences:
        for token in sentence.tokens:
            if token.text.lower() not in en_sw:
                count += 1
    return count
```

## **Part-of-speech tagging

Grammatical structure (verbs, adverbs, nouns, adjectives). 

```
en = snlp.Pipeline(lang='en')

txt = "Yo you around? A friend of mine's lookin."
pos = en(txt)

# word/POS
def print_pos(doc):
    text = ""
        for sentence in doc.sentences:
            for token in sentence.tokens:
                text += token.words[0].text + "/" + token.words[0].upos + " "
            text += "\n"
    return text

def word_counts_v3(x, pipeline=en):
    doc = pipeline(x)
    totals = 0.
    count = 0.
    non_word = 0.
    for sentence in doc.sentences:
        totals += len(sentence.tokens) # (1)
        for token in sentence.tokens:
            if token.text.lower() not in en_sw:
                if token.words[0].upos not in ['PUNCT', 'SYM']:
                    count += 1.
                else:
                    non_word += 1.
    non_word = non_word / totals
    return pd.Series([count, non_word], index=['Words_NoPunct', 'Punct'])
```

## **Stemming and Lematization**

> depend: depends, depending, depended, dependent

```
lemmas = ""
for sentence in lemma.sentences:
    for token in sentence.tokens:
        lemmas += token.words[0].lemma +"/" + token.words[0].upos + " "
    lemmas += "\n"
print(lemmas)
```

## **Vectorizing text**

First issue, arbitrary lengths

Second issue, representation of words with a numerical quantity or feature


## Count-based vectorization

```
from sklearn.feature_extraction.text import CountVectorizer
```


## TF-IDF

Term Frequency-Inverse Document Frequency

$$TF-IDF(t, d)=TF(t,d)*IDF(t)$$

```
from sklearn.feature_extraction.text import TfidfTransformer
```

## Word Vectors

OR Embeddings. A representation of a word in some vector space, and the word can be considered embedded in that space. 

Core hypothesis: words that occur near each other are related to each other. 

continuous bag-of-words / continuous skip-gram

```
# Pretrained models
!pip install gensim

from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

model_w2v = api.load("word2vec-google-news-300")

model_w2v.most_similar("cookies",topn=10)
model_w2v.doesnt_match(["USA","Canada","India","Tokyo"])
```