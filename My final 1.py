#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import nltk
#import spacy
import re
import string,unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from py_lex import EmoLex
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns


# In[3]:


df1= pd.read_csv("metadata_nyc.csv")
df1.head()


# In[4]:


df2= pd.read_csv("reviewContentNYC.csv")
df2.head()


# In[5]:


df1["review"]=df2["review"]


# In[6]:


columntitles=["date","user_id","prod_id","rating","review","label"]
df1=df1.reindex(columns=columntitles)
df= df1
df.head()


# In[7]:


df.shape


# In[9]:


df["label"].value_counts()              


# In[10]:


df["label"]=df["label"].replace(-1,0)                #1 authentic     #0 fake
df["label"].value_counts()


# In[11]:


ros = RandomUnderSampler(random_state=777)
dfx, dfy = ros.fit_sample(df.iloc[:,0:5], df.label)
print(dfx.shape)
df = pd.concat([dfx, dfy], axis = 1)
df['label'].value_counts()


# In[12]:


df.shape                                                                                                           #upto here this file is saved as "mappeddata_NYC.csv" for future reference


# In[13]:


#1 Average User rating
temp= df.groupby(['user_id'])['rating'].agg(np.mean).reset_index()
temp.rename(columns={'rating':'avg_Urating'}, inplace=True)
df = pd.merge(df, temp, how='outer', on=['user_id'])


# In[14]:


#2 Average Product rating
temp= df.groupby(['prod_id'])['rating'].agg(np.mean).reset_index()
temp.rename(columns={'rating':'avg_Prating'}, inplace=True)
df = pd.merge(df, temp, how='outer', on=['prod_id'])


# In[15]:


#3 No. of reviews by a user
temp = df['user_id'].value_counts().rename_axis('user_id').to_frame('UCcounts')
df = pd.merge(df, temp, how='outer', on=['user_id'])


# In[16]:


#4 No. of reviews on a product
temp = df['prod_id'].value_counts().rename_axis('prod_id').to_frame('PCcounts')
df = pd.merge(df, temp, how='outer', on=['prod_id'])


# In[18]:


#5 Number of words(review length)
df['#ofwords'] = df['review'].str.count(' ') + 1
#6 Avg word count in user's review
temp = df.groupby(['user_id'])['#ofwords'].agg(np.mean).reset_index()
temp.rename(columns={'#ofwords':'Uavg#word'}, inplace=True)
df = pd.merge(df, temp, how='outer', on=['user_id'])


# In[19]:


#7 Rating deviation                                      #Mine 1
df["ratdev"]=(df["avg_Prating"]-df1["rating"]).abs()


# In[20]:


df['postags'] = nltk.tag.pos_tag_sents(df['review'].apply(nltk.word_tokenize).tolist())                                    #took 18 min


# In[21]:


def NounCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("NN"):
            nouns.append(word)
    return nouns

df["nouns"] = df["postags"].apply(NounCounter)
df["noun_count"] = df["nouns"].str.len()


def VerbCounter(x):
    verbs = []
    for (word, pos) in x:
        if pos.startswith("VB"):
            verbs.append(word)
    return verbs

df["verbs"] = df["postags"].apply(VerbCounter)
df["verb_count"] = df["verbs"].str.len()


def AdverbCounter(x):
    adverbs = []
    for (word, pos) in x:
        if pos.startswith("RB"):
            adverbs.append(word)
    return adverbs

df["adverbs"] = df["postags"].apply(AdverbCounter)
df["adverb_count"] = df["adverbs"].str.len()


def AdjCounter(x):
    adjs = []
    for (word, pos) in x:
        if pos.startswith("JJ"):
            adjs.append(word)
    return adjs

df["adjectives"] = df["postags"].apply(AdjCounter)
df["adj_count"] = df["adjectives"].str.len()


# In[22]:


#8 noun%                                                        #Mine 2
df["noun%"]= (df["noun_count"]/df["#ofwords"])*100


# In[23]:


#9 imaginative to informative ratio                              #Mine 3
df["imag_to_info"]= (df["adverb_count"]+df["verb_count"])/(df["noun_count"]+df["adj_count"])


# In[24]:


#10 self reference diversity                                     #Mine 4
def PronounCounter(x):
    pronouns = []
    for (word, pos) in x:
        if pos.startswith("PRP"):
            pronouns.append(word)
    return pronouns

df["pronouns"] = df["postags"].apply(PronounCounter)
df["pronoun_count"] = df["pronouns"].str.len()


def PossPronounCounter(x):
    posspronouns = []
    for (word, pos) in x:
        if pos.startswith("PRP$"):
            posspronouns.append(word)
    return posspronouns

df["posspronouns"] = df["postags"].apply(PossPronounCounter)
df["posspronoun_count"] = df["posspronouns"].str.len()

df["perprocount"]= df["pronoun_count"]-df["posspronoun_count"]                 
df["selfreference_div"]= df["perprocount"]/df["pronoun_count"]                                                         #remember to change empty valus to 0 in this column


# In[25]:


#11 capital counts                                               #Mine 5
def tokens(text):
    return  str(text).split() 
df["tokens"]= df["review"].apply(tokens)

def countcapital(wordarray):
    count=0
    for word in wordarray:
        if(word.isupper()):
            count=count+1
    return count    

df["capitals"]= df["tokens"].apply(countcapital)


# In[26]:


df.head()


# In[27]:


#12 assumption and future words count                            #Mine 6
  
from nltk.corpus import wordnet                                                                                        #took 2 mins  # means pos_tag takes time
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = text
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df["text_lemmatized"] = df["postags"].apply(lemmatize_words)

df['lemmatizedtokens'] = df['text_lemmatized'].apply(nltk.word_tokenize).tolist()  

assum_fut=["assume","feel","may","think","believe","wonder","will","shall","guess"] 

def count(array):
    count=0
    for word in array:
        if word in assum_fut:
            count=count+1;
    return count 

df["assum_fut_count"]= df["lemmatizedtokens"].apply(count)


# In[28]:


df.head()


# In[33]:


df.to_csv('halfwaydone.csv')


# In[ ]:





# In[2]:


df= pd.read_csv('halfwaydone.csv')


# In[3]:


#13 #of reviews posted by a reviewer on a particular date               #Mine 7
df["ones"]= 1
temp = df.groupby(['date','user_id'])['ones'].agg(np.count_nonzero).reset_index()
temp.rename(columns={'ones':'U_rev_perday'}, inplace=True)
df = pd.merge(df, temp, how='outer', on=['date','user_id'])


# In[5]:


#14 of reviews posted by a reviewer on a particular product -- multiple reviews             #Mine 8
temp = df.groupby(['prod_id','user_id'])['ones'].agg(np.count_nonzero).reset_index()
temp.rename(columns={'ones':'mul_rev'}, inplace=True)
df = pd.merge(df, temp, how='outer', on=['prod_id','user_id'])


# In[6]:


#15 positive to negative ratio of a user                                   #Mine 9

#4ormore
temp= df.groupby(['user_id'])['rating'].agg(lambda val: (val >=4 ).sum()).reset_index()       
temp.rename(columns={'rating':'4ormore'}, inplace=True)
df = pd.merge(df, temp, how='outer', on=['user_id'])

#2orless
temp= df.groupby(['user_id'])['rating'].agg(lambda val: (val <=2 ).sum()).reset_index()        
temp.rename(columns={'rating':'2orless'}, inplace=True)
df = pd.merge(df, temp, how='outer', on=['user_id'])

df["pos_to_neg_ratio"]= df["4ormore"]/df["2orless"]


# In[9]:


#16 Subjectivity 
df['review'] = df.review.astype(str)
df['subjectivity'] = df['review'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)


# In[13]:


#17 Polarity
df['review'] = df.review.astype(str)
df['polarity'] = df['review'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)


# In[14]:


#18 Rating
#handling missing and inf values in self ref diversity and positive to neg ratio column
df=df.replace([np.inf, -np.inf], np.nan)                  #all inf or -inf replaced by nan in whole dataset

mean= df["selfreference_div"].mean()
df["selfreference_div"].fillna(mean,inplace=True)

mean=df["pos_to_neg_ratio"].mean()
df["pos_to_neg_ratio"].fillna(mean,inplace=True)


# In[15]:


df.columns


# # Text preprocessing

# In[16]:


'''Lowercasing'''
df["reviewcopy"]=df["review"]
df["reviewcopy"] = df["reviewcopy"].str.lower()

'''Removing Punctuations'''
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["reviewcopy"] = df["reviewcopy"].apply(lambda text: remove_punctuation(text))


'''Removing stopwords'''
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["reviewcopy"] = df["reviewcopy"].apply(lambda text: remove_stopwords(text))


'''Removal of URL'''
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
df["reviewcopy"] = df["reviewcopy"].apply(lambda text: remove_urls(text))


# In[17]:


#19 lexical Density
df['#ofwordswostp'] = df['reviewcopy'].str.count(' ') + 1
df['le_d'] = (df['#ofwordswostp']/df['#ofwords'])*100


# In[18]:


df.to_csv('fullwaydone.csv')


# In[19]:


df.columns


# In[20]:


#dropping unnecessary columns
df.drop(['Unnamed: 0','date','user_id','prod_id','review','postags','nouns','noun_count','verbs','verb_count','adverbs', 'adverb_count', 'adjectives', 'adj_count','pronouns', 'pronoun_count', 'posspronouns',
       'posspronoun_count', 'perprocount','tokens','text_lemmatized', 'lemmatizedtokens','ones','reviewcopy',
       '#ofwordswostp'],axis=1,inplace=True)


# In[24]:


df.columns


# In[ ]:




