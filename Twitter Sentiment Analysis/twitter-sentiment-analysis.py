#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import string
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[3]:


import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# In[4]:


from nltk import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words


# In[5]:


import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.models import KeyedVectors


# In[6]:


from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy


# In[7]:


import joblib


# In[8]:


from collections import Counter


# In[9]:


pio.renderers.default = "notebook"


# ## Read Data

# In[11]:


data = pd.read_csv("data/training.16m.tweet.csv", encoding="ISO-8859-1", names=["label", "id", "date", "query", "user", "tweet"])


# In[12]:


data.sample(5)


# In[13]:


data.describe()


# In[14]:


data.info()


# In[ ]:





# ### Clean tweets: remove @username or https website from text.

# In[10]:


def clean_tweets_user(text):
    REGx_tweet = r"@\S+|https?:\S+|http?:\S"
    text = re.sub(REGx_tweet, "", str(text).lower()).strip()
    return "".join(text)


# #### remove hashtag

# In[ ]:


def remove_hashtag(text) -> str:
    REGx_hastag = r"#[A-Za-z0-9_]+"
    text = re.sub(REGx_hastag, " ", str(text).lower()).strip()
    return text


# ### Clean punctuation: remove punctuation from text.

# In[11]:


def clean_punctuation(text):
    clean_text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    return "".join(clean_text.strip())


# ### Clean number from text.

# In[12]:


def clean_numbers(text):
    clean_text = re.sub('\w*\d\w*', "", text)
    return "".join(clean_text)
    


# ### Clean stop words.

# In[13]:


stop_words = stopwords.words('english')
stop_words.remove("not")
def clean_stopwords(text):
    wordList = word_tokenize(text)
    clean_text = []
    for word in wordList:
            if word in stop_words:
                continue
            else:
                clean_text.append(word+" ")
    return "".join(clean_text)


# ### Stemmers & Lemmatized .

# In[14]:


snowBallStemmer = SnowballStemmer("english")
def tweet_stemmers(text):
    wordList = word_tokenize(text)
    stemWords = [snowBallStemmer.stem(word+" ") for word in wordList]
    return "".join(stemWords)


# In[15]:


get_ipython().run_cell_magic('time', '', 'nlp = spacy.load("en_core_web_lg",disable = [\'tagger\',\'perser\',\'ner\'])\ndef tweet_lemmatized(text) -> str:\n    doc = nlp(text)\n    clean_text =[str(word.lemma_) if word.lemma_ != "-PRON-" else str(word) for word in doc]\n    return " ".join(clean_text)')


# ### Remove emojies

# In[16]:


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r" ", text)


# #### remove arabic word 

# In[ ]:


def remove_arabic(text) -> str:
    arabic_compile = re.compile(r"[\u0600-\u06FF]+",flags=re.UNICODE )
    clean_text = arabic_compile.sub(r" ", text)
    clean_text = clean_text.strip()
    clean_text = clean_text.encode("ascii", "ignore")
    return str(clean_text.decode())


# ### Encode Text

# In[17]:


def encode_text(text):
    clean_text = text.strip()
    clean_text = clean_text.encode("ascii", "ignore")
    return str(clean_text.decode())


# ## Pre Process Text

# In[22]:


get_ipython().run_cell_magic('time', '', 'data["tweet"] = data["tweet"].apply(lambda text: clean_tweets_user(text))')


# In[23]:


get_ipython().run_cell_magic('time', '', 'data["tweet"] = data["tweet"].apply(lambda text: tweet_lemmatized(text))')


# In[24]:


get_ipython().run_cell_magic('time', '', 'data["tweet"] = data["tweet"].apply(lambda text: tweet_stemmers(text))')


# In[25]:


get_ipython().run_cell_magic('time', '', 'data["tweet"] = data["tweet"].apply(lambda text: clean_punctuation(text))')


# In[26]:


get_ipython().run_cell_magic('time', '', 'data["tweet"] = data["tweet"].apply(lambda text: clean_stopwords(text))')


# In[27]:


get_ipython().run_cell_magic('time', '', 'data["tweet"] = data["tweet"].apply(lambda text: clean_numbers(text))')


# In[28]:


get_ipython().run_cell_magic('time', '', 'data["tweet"] = data["tweet"].apply(lambda text: encode_text(text))')


# ## Gensim Model
# #####   

# #### Cpu Cores count 

# In[30]:


import multiprocessing
cores = multiprocessing.cpu_count()
cores


# #### TaggedDocument

# In[31]:


get_ipython().run_cell_magic('time', '', 'tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data["tweet"])]')


# In[32]:


tagged_data_DBOW, tagged_data_DM = train_test_split(tagged_data, test_size=0.5,random_state=42)


# #### Train Model

# #### PV-DBOW model

# In[33]:


get_ipython().run_cell_magic('time', '', 'model_dbow = gensim.models.doc2vec.Doc2Vec(tagged_data_DBOW ,vector_size=300, negative=10, workers=(cores-1),\n                                           window=5,alpha=0.025, min_alpha=0.0001, seed=1, min_count=5,\n                                           sample=0.001, epochs=65,hs=0,dm=0, ns_exponent=0.75)')


# In[34]:


get_ipython().run_cell_magic('time', '', 'model_dbow.build_vocab(tagged_data_DBOW, update=True)')


# In[35]:


get_ipython().run_cell_magic('time', '', 'model_dbow.train(tagged_data_DBOW, total_examples=len(tagged_data_DBOW), epochs=65)')


# In[36]:


model_dbow_file = "models/DBOW_model/"


# In[37]:


get_ipython().run_cell_magic('time', '', 'model_dbow.save(model_dbow_file+"model.model")')


# In[38]:


model_dbow.wv.save_word2vec_format(model_dbow_file+"model_format/model.bin")


# In[39]:


model_dbow.wv.save_word2vec_format(model_dbow_file+"model_format/model.csv", binary=False)


# In[40]:


model_dbow.wv.save_word2vec_format(model_dbow_file+"model_format/model.txt", binary=False)


# #### DM Model

# In[41]:


get_ipython().run_cell_magic('time', '', 'model_dm = gensim.models.doc2vec.Doc2Vec(tagged_data_DM ,vector_size=300, negative=10, workers=(cores-1),\n                                         window=5,alpha=0.025, min_alpha=0.0001, seed=1, min_count=5,\n                                         sample=0.001, epochs=65,hs=0, dm=1, ns_exponent=0.75)')


# In[42]:


get_ipython().run_cell_magic('time', '', 'model_dm.build_vocab(tagged_data_DM, update=True)')


# In[43]:


get_ipython().run_cell_magic('time', '', 'model_dm.train(tagged_data_DM, total_examples=len(tagged_data_DM), epochs=65)')


# In[44]:


model_dm_file = "models/DM_model/"


# In[45]:


get_ipython().run_cell_magic('time', '', 'model_dm.save(model_dm_file+"model.model")')


# In[47]:


model_dm.wv.save_word2vec_format(model_dm_file+"model_format/model.bin")


# In[48]:


model_dm.wv.save_word2vec_format(model_dm_file+"model_format/model.csv", binary=False)


# In[49]:


model_dm.wv.save_word2vec_format(model_dm_file+"model_format/model.txt", binary=False)


# ### Load Models

# In[18]:


doc2vec_dm_model = gensim.models.doc2vec.Doc2Vec.load("models/DM_model/model.model")


# In[19]:


doc2vec_dm_model.wv.most_similar("text")


# In[20]:


len(doc2vec_dm_model.wv.vocab)


# In[21]:


doc2vec_dbow_model = gensim.models.doc2vec.Doc2Vec.load("models/DBOW_model/model.model")


# In[22]:


doc2vec_dbow_model.wv.most_similar("great")


# In[23]:


len(doc2vec_dbow_model.wv.vocab)


# ### Tokenizer spacy

# In[24]:


nlp = English()
tokenizer_ng = Tokenizer(nlp.vocab)


# ## Train data for ML 

# In[25]:


train = pd.read_csv("data/data_tweet.csv")


# In[26]:


train.sample(5)


# In[27]:


train.info()


# In[28]:


train = train.drop("Unnamed: 0", axis=1)


# In[29]:


train["tweet"] = train["tweet"].apply(lambda text: clean_tweets_user(text))

train["tweet"] = train["tweet"].apply(lambda text: remove_emoji(text))
train["tweet"] = train["tweet"].apply(lambda text: encode_text(text))

train["tweet"] = train["tweet"].apply(lambda text: tweet_lemmatized(text))
train["tweet"] = train["tweet"].apply(lambda text: tweet_stemmers(text))

train["tweet"] = train["tweet"].apply(lambda text: clean_numbers(text))
train["tweet"] = train["tweet"].apply(lambda text: clean_punctuation(text))
train["tweet"] = train["tweet"].apply(lambda text: clean_stopwords(text))


# ### Doc2Vec & ML Models

# In[30]:


tagged_train_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(train["tweet"])]


# In[31]:


tagged_data_train_DBOW, tagged_data_train_DM = train_test_split(tagged_train_data, test_size=0.5,random_state=42)


# In[32]:


doc2vec_dbow_model.build_vocab(tagged_data_train_DBOW, update=True)


# In[33]:


doc2vec_dm_model.build_vocab(tagged_data_train_DM, update=True)


# In[34]:


doc2vec_dm_model.estimate_memory()


# In[35]:


doc2vec_dbow_model.estimate_memory()


# In[36]:


from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
new_model = ConcatenatedDoc2Vec([doc2vec_dbow_model, doc2vec_dm_model])


# In[37]:


from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
train[["label"]] = enc.fit_transform(train[["sentiment"]])


# In[38]:


train_x, test_x = train_test_split(train, test_size=0.2, random_state=42)
print("Train size:", len(train_x))
print("Test size:", len(test_x))


# In[39]:


len(doc2vec_dbow_model.wv.vocab)


# In[40]:


len(doc2vec_dm_model.wv.vocab)


# ## Vector of word

# In[41]:


get_ipython().run_cell_magic('time', '', "train_a = []\nfor text in train_x['tweet']:\n    train_a.append(new_model.infer_vector([str(word) for word in tokenizer_ng(text)]))   ")


# In[42]:


train_vec = pd.DataFrame(train_a)


# In[43]:


get_ipython().run_cell_magic('time', '', "test_a=[]\nfor text in test_x['tweet']:\n    test_a.append(new_model.infer_vector([str(word) for word in tokenizer_ng(text)]))")


# In[44]:


test_vec = pd.DataFrame(test_a)


# In[45]:


y_train = train_x["label"]
y_test = test_x["label"]


# ### Scale data

# In[46]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()


# In[47]:


train_scale = std.fit_transform(train_vec)
test_scale = std.fit_transform(test_vec)


# In[48]:


train_norm = pd.DataFrame(train_scale) 
test_norm = pd.DataFrame(test_scale) 


# ### ML Model

# In[49]:


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier


# In[50]:


get_ipython().run_cell_magic('time', '', 'ovr_sv = OneVsRestClassifier(SVC(kernel="rbf", C=20))\novr_sv.fit(train_norm, y_train)\novr_sv_accuracy = round(ovr_sv.score(test_norm, y_test) * 100, 3)\novr_sv_accuracy')


# ### Fine tuniing

# In[53]:


# Grid Search
param_grid = {"C": [10,20,30]}


# In[54]:


get_ipython().run_cell_magic('time', '', 'svc_g = OneVsRestClassifier(SVC(kernel="rbf"))\ngrid_svc = GridSearchCV(estimator=svc_g, param_grid=param_grid, cv=5)')


# In[55]:


get_ipython().run_cell_magic('time', '', 'grid_svc.fit(train_norm, y_train)')


# In[56]:


get_ipython().run_cell_magic('time', '', 'print(" Results from Grid Search ")\nprint("\\n The best estimator across ALL searched params:\\n", grid_svc.best_estimator_)\nprint("\\n The best score across ALL searched params:\\n", grid_svc.best_score_)\nprint("\\n The best parameters across ALL searched params:\\n", grid_svc.best_params_)')


# In[ ]:





# ### Metric

# In[50]:


loaded_model = joblib.load("models/ml_model/ml_model.sav")


# In[51]:


svc_pre = ovr_sv.score(test_norm, y_test)


# In[57]:


get_ipython().run_cell_magic('time', '', 'predictions = cross_val_predict(ovr_sv, test_norm, y_test, cv=15)\nprint(predictions)')


# In[57]:


print("Classification Report:\n", classification_report(y_test,predictions))


# In[58]:


print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


# ### save ml model

# In[59]:


filename = "models/ml_model/ml_model.sav"
joblib.dump(ovo_sv, filename)


# In[61]:


loaded_model = joblib.load(filename)
result = loaded_model.score(test_norm, y_test)
print(result)

