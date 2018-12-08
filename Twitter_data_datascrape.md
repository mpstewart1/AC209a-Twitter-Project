


```python
# !pip install tweepy
# !pip install nltk
# !pip install twython
# !pip install jsonpickle
# !pip install botometer
import csv
```




```python
import sys
import jsonpickle
import os
import tweepy
import nltk
import pandas as pd
import json
from pandas.io.json import json_normalize
from datetime import datetime, timezone
import numpy as np
import botometer
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly

import warnings
warnings.filterwarnings("ignore")
plotly.offline.init_notebook_mode()
```




```python
auth = tweepy.OAuthHandler('pr0AH7Ot5sZmig4u3bA6j51ty', 'tNteF0tRlEjKJfkkWQaIv5myqT9oBqrIVOYPQJOMjBTJhn9SAF')
auth.set_access_token('934846563825930241-yO5rosUB4x8eFMO0J7IXV1UZM0RzbgL', 'CbqfvlRonXo2JiIyxqCqeZynwkslNcDPmGFQ9KBEh8Mch')
api = tweepy.API(auth)

mashape_key = "uIX3UUkrh7mshux9VLXhN1FcUYY0p1ZEJpCjsnCHKddXFfIzhf"
twitter_app_auth = {
    'consumer_key': 'pr0AH7Ot5sZmig4u3bA6j51ty',
    'consumer_secret': 'tNteF0tRlEjKJfkkWQaIv5myqT9oBqrIVOYPQJOMjBTJhn9SAF',
    'access_token': '934846563825930241-yO5rosUB4x8eFMO0J7IXV1UZM0RzbgL',
    'access_token_secret': 'CbqfvlRonXo2JiIyxqCqeZynwkslNcDPmGFQ9KBEh8Mch',
  }
```




```python
"""
#Botometer

bom = botometer.Botometer(wait_on_ratelimit=True,mashape_key=mashape_key,**twitter_app_auth)
# Check a single account by screen name
result1 = bom.check_account('@clayadavis')

# Check a single account by id
result2 = bom.check_account(1548959833)

"""
```




```python
# Social Honeypot Dataset

# Legitimate user info
lu_df = pd.read_csv('legitimate_users.txt', sep = '\t', header = None)
lu_df.columns = ['UserID', 'CreatedAt', 'CollectedAt', 'NumerOfFollowings', 'NumberOfFollowers', 'NumberOfTweets', 'LengthOfScreenName', 'LengthOfDescriptionInUserProfile']
# Content polluters info
bots_df = pd.read_csv('bot_users.txt', sep = '\t', header = None)
bots_df.columns = ['UserID', 'CreatedAt', 'CollectedAt', 'NumerOfFollowings', 'NumberOfFollowers', 'NumberOfTweets', 'LengthOfScreenName', 'LengthOfDescriptionInUserProfile']
```




```python
lu_list = lu_df['UserID'].values.astype(int)
bot_list = bots_df['UserID'].values.astype(int)
lu_df.head()
```




```python
# Given a name list and number of tweets needed to extract for each account
# Return a dictionary of dataframes
# Each dataframe contains info of one user
def API_scrap(name_list, count_num):
    fail_lst = []
    user_dfs = {}
    for name in name_list:
        print(name)
        try:
            status_a = api.user_timeline(name, count = count_num, tweet_mode = 'extended')
            user_dfs[name] = pd.DataFrame()
            for i in range(len(status_a)):
                json_str = json.dumps(status_a[i]._json)
                jdata = json_normalize(json.loads(json_str))
                user_dfs[name] = user_dfs[name].append(jdata, ignore_index=True)

        except:
            fail_lst.append(name)
            continue
    
    return user_dfs, fail_lst
```




```python
user_dfs, fail_lst = API_scrap(lu_list[0:100], 200)
```




```python
successful_users = [name for name in user_dfs.keys() if name not in fail_lst]
successful_users
```




```python
user_dfs[successful_users[0]].columns[0:40]
```




```python
single_user = user_dfs[successful_users[0]][['full_text','created_at',
                                             'entities.user_mentions']].copy().reset_index()
single_user['user_id'] = successful_users[0]
single_user.drop(['index'], inplace=True, axis = 1)

display(single_user)
```




```python
for i, name in enumerate(successful_users):
    try:
        single_user = user_dfs[successful_users[i]][['full_text','created_at','entities.user_mentions']].copy().reset_index()
        single_user['user_id'] = successful_users[i]
        single_user.drop(['index'], inplace=True, axis = 1)
        filepath ='data_NLP/bots/{}_tweets.csv'.format(name)
        single_user.to_csv(filepath, sep='\t')
    except:
        print("couldnt do user {} for some reason".format(successful_users[i]))
```




```python
# get some bots! 
bot_dfs, bot_fail_lst = API_scrap(bot_list[0:100], 200)
```




```python
successful_users_bots = [name for name in bot_dfs.keys() if name not in bot_fail_lst]
successful_users_bots
```




```python
for i, name in enumerate(successful_users_bots):
    try:
        single_user = bot_dfs[successful_users_bots[i]][['full_text','created_at','entities.user_mentions']].copy().reset_index()
        single_user['user_id'] = successful_users_bots[i]
        single_user.drop(['index'], inplace=True, axis = 1)
        filepath ='data_NLP/bots/{}_tweets.csv'.format(name)
        single_user.to_csv(filepath, sep='\t')
    except:
        print("couldnt do user {} for some reason".format(successful_users_bots[i]))
```


    couldnt do user 8377772 for some reason
    couldnt do user 7721172 for some reason
    



```python
from os import listdir
from os.path import isfile, join
!ls
mypath_bots = 'data_NLP/bots/'
mypath_legit = 'data_NLP/legit/'
botfiles = [f for f in listdir(mypath_bots) if isfile(join(mypath_bots, f)) and not f=='.DS_Store']
legitfiles = [f for f in listdir(mypath_bots) if isfile(join(mypath_bots, f))and not f=='.DS_Store']
```


    AM207_HW9_2018_Matthew-Stewart.ipynb [31mbot_users.txt[m[m
    BarackObama_tweets.csv               [34mcresci-2017.csv (1)[m[m
    LICENSE                              [34mdata_NLP[m[m
    NLP_EDA.ipynb                        elonmusk_tweets.csv
    README.md                            [31mlegitimate_users.txt[m[m
    Twitter_data_datascrape.ipynb        tweets.json
    _config.yml




```python
# test code on one dataframe 
tweets_df = pd.DataFrame.from_csv('data_NLP/bots/' + botfiles[0],sep='\t')
tweets_df.rename(index=str, columns={"full_text": "text"}, inplace=True)

tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])
```




```python
def create_NLP_dataframe(list_of_users, botBool):
    
    tweets_df = pd.DataFrame.from_csv('data_NLP/bots/' + list_of_users[0] ,sep='\t')
    tweets_df.rename(index=str, columns={"full_text": "text"}, inplace=True)

    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])
    
    if botBool:
        tweets_df['botBool'] = 1
    else:
        tweets_df['botBool'] = 0

    # number of hashtags 
    tweets_df['num_hashtags'] = tweets_df['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

    # number of all-caps words 
    tweets_df['num_upper'] = tweets_df['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

    # deal with emojis
    import emoji

    class Emoticons:
        POSITIVE = ["*O", "*-*", "*O*", "*o*", "* *",
                    ":P", ":D", ":d", ":p",
                    ";P", ";D", ";d", ";p",
                    ":-)", ";-)", ":=)", ";=)",
                    ":<)", ":>)", ";>)", ";=)",
                    "=}", ":)", "(:;)",
                    "(;", ":}", "{:", ";}",
                    "{;:]",
                    "[;", ":')", ";')", ":-3",
                    "{;", ":]",
                    ";-3", ":-x", ";-x", ":-X",
                    ";-X", ":-}", ";-=}", ":-]",
                    ";-]", ":-.)",
                    "^_^", "^-^"]

        NEGATIVE = [":(", ";(", ":'(",
                    "=(", "={", "):", ");",
                    ")':", ")';", ")=", "}=",
                    ";-{{", ";-{", ":-{{", ":-{",
                    ":-(", ";-(",
                    ":,)", ":'{",
                    "[:", ";]"
                    ]

    def getPositiveTweetEmojis(tweet):
        return ''.join(c for c in tweet if c in Emoticons.POSITIVE)

    def getNegativeTweetEmojis(tweet):
        return ''.join(c for c in tweet if c in Emoticons.NEGATIVE)

    def extractAllEmojis(str):
        return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

    # all emojis in a text 
    def extract_emojis(str):
        return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)
    tweets_df['all_emojis'] = [extractAllEmojis(tweet) for tweet in tweets_df['text']]
    tweets_df['positive_emojis'] = [getPositiveTweetEmojis(tweet) for tweet in tweets_df['text']]
    tweets_df['negative_emojis'] = [getNegativeTweetEmojis(tweet) for tweet in tweets_df['text']]

    # clean tweets 
    from bs4 import BeautifulSoup
    tweets_df['text'] = [re.sub(r'http[A-Za-z0-9:/.]+','',str(tweets_df['text'][i])) for i in range(len(tweets_df['text']))]
    removeHTML_text = [BeautifulSoup(tweets_df.text[i], 'lxml').get_text() for i in range(len(tweets_df.text))]
    tweets_df.text = removeHTML_text
    tweets_df['text'] = [re.sub(r'@[A-Za-z0-9]+','',str(tweets_df['text'][i])) for i in range(len(tweets_df['text']))]

    weird_characters_regex = re.compile(r"[^\w\d ]")
    tweets_df.text = tweets_df.text.str.replace(weird_characters_regex, "")
    RT_bool = [1 if text[0:2]=='RT' else 0 for text in tweets_df['text']]
    tweets_df['RT'] = RT_bool
    tweets_df.text = tweets_df.text.str.replace('RT', "")

    # average time between retweets
    retweet_table = tweets_df[['created_at','RT']].copy()
    retweet_table  = retweet_table[retweet_table.RT == 1]
    total_observation_period_rt = retweet_table['created_at'][0]-retweet_table['created_at'][-1]
    time_between_average_rt = float(len(retweet_table))/total_observation_period_rt.days

    # average number of mentions
    tweets_df['num_mentions'] = [len(eval(tweets_df['entities.user_mentions'][i])) for i in range(len(tweets_df.text))]

    # average time between mentions
    mention_table = tweets_df[['num_mentions','created_at']].copy()
    mention_table  = mention_table[mention_table['num_mentions']>0]
    mention_table.head()
    total_observation_period_mention = mention_table['created_at'][0]-mention_table['created_at'][-1]
    time_between_average_mention = float(len(mention_table))/total_observation_period_mention.days
    #print(time_between_average_rt,time_between_average_mention)
    
    # get word count, char count
    tweets_df['word_count'] = tweets_df['text'].apply(lambda x: len(str(x).split(" ")))
    tweets_df['char_count'] = tweets_df['text'].str.len() ## this also includes spaces
    #display(tweets_df[['text','char_count','word_count']].head())


    # build some average features 
    tweets_df['avg_time_between_rt'] = time_between_average_rt
    tweets_df['avg_time_between_mention'] = time_between_average_mention
    tweets_df['avg_num_mentions'] = np.mean(tweets_df['num_mentions'])
    tweets_df['avg_num_hashtags'] = np.mean(tweets_df['num_hashtags'])
    tweets_df['avg_num_caps'] = np.mean(tweets_df['num_upper'])
    tweets_df['avg_words_per_tweet'] = np.mean(tweets_df['word_count'])
    tweets_df['emoji_bool'] = [1 if len(tweets_df['all_emojis'][i])>0 else 0 for i in range(len(tweets_df))]
    tweets_df['emoji_p_bool'] = [1 if len(tweets_df['positive_emojis'][i])>0 else 0 for i in range(len(tweets_df))]
    tweets_df['emoji_n_bool'] = [1 if len(tweets_df['negative_emojis'][i])>0 else 0 for i in range(len(tweets_df))]
    tweets_df['emoji_pn_bool'] = [1 if len(tweets_df['positive_emojis'][i])>0 and 
                                  len(tweets_df['negative_emojis'][i])>0 else 0 for i in range(len(tweets_df))]
    tweets_df['percent_with_emoji'] = np.mean(tweets_df['emoji_bool'])
    tweets_df['percent_with_p_emoji'] = np.mean(tweets_df['emoji_p_bool'])
    tweets_df['percent_with_n_emoji'] = np.mean(tweets_df['emoji_n_bool'])
    tweets_df['percent_with_pn_emoji'] = np.mean(tweets_df['emoji_pn_bool'])


    # check data types 
    display(tweets_df.dtypes)

    # get average word length
    def avg_word(sentence):
        words = sentence.split()
        if len(words) == 0:
            return 0
        return (sum(len(word) for word in words)/len(words))

    tweets_df['avg_word'] = tweets_df['text'].apply(lambda x: avg_word(x))
    tweets_df[['text','avg_word']].head()

    # all lowercase 
    tweets_df['text'] = tweets_df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    display(tweets_df['text'].head())

    # remove stopwords and punctuation
    from nltk.corpus import stopwords
    from string import punctuation

    retweet = ['RT','rt']
    stoplist = stopwords.words('english') + list(punctuation) + retweet
    #stop = stopwords.words('english')
    tweets_df['text'] = tweets_df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stoplist))
    tweets_df['text'].head()

    # add sentiment feature
    from textblob import Word, TextBlob  
    tweets_df['sentiment'] = tweets_df['text'].apply(lambda x: TextBlob(x).sentiment[0])
    tweets_df['polarity'] = tweets_df['text'].apply(lambda x: TextBlob(x).sentiment[1])
    tweets_df = tweets_df.sort_values(['sentiment'])

    # add list of nouns
    import nltk

    tweets_df['nouns'] = [TextBlob(tweets_df.text[i]).noun_phrases for i in range(len(tweets_df.text))]

    tweets_df['POS_tag_list'] = [TextBlob(tweets_df.text[i]).tags for i in range(len(tweets_df.text))]
    tweets_df['POS_tag_list'] = [[tuple_POS[1] for tuple_POS in tweets_df['POS_tag_list'][i]] for i in range(len(tweets_df.text))]

    # look at 10 most frequent words
    freq = pd.Series(' '.join(tweets_df['text']).split()).value_counts()[:20]

    tweets_df['top_20_nouns'] = [freq.index.values for i in range(len(tweets_df))]

    list_of_POS = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS",
    "PDT","POS","PRP","PRP","RB","RBR","RBS","RP","TO","UH","VB","VBD","VBG","VBN",
    "VBP","VBZ","WDT","WP","WP$","WRB"]

    for POS in list_of_POS:
        varname = POS+"_count"
        tweets_df[varname] = [tweets_df['POS_tag_list'][i].count(POS) for i in range(len(tweets_df.text))]
        
    # get full text in a string
    full_tweet_text = ""
    for i in range(len(tweets_df)):
        full_tweet_text = full_tweet_text + tweets_df['text'][i]
    full_tweet_text_list = full_tweet_text.split()
    unique_full_text = len(set(full_tweet_text_list))
        
    # features based on full text 
    tweets_df['word_diversity'] = unique_full_text/len(full_tweet_text)
    tweets_df['overall_sentiment'] = TextBlob(full_tweet_text).sentiment[0]
    tweets_df['overall_polarity'] = TextBlob(full_tweet_text).sentiment[1]
        
    subset_df= tweets_df[['word_diversity','overall_sentiment','overall_polarity']]
    return subset_df.iloc[0]

```




```python
bot_data = [create_NLP_dataframe(botfiles[i], 1) for i in range(len(botfiles))]
```



    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-374-af3544a06216> in <module>()
    ----> 1 bot_data = [create_NLP_dataframe(botfiles[i], 1) for i in range(len(botfiles))]
    

    <ipython-input-374-af3544a06216> in <listcomp>(.0)
    ----> 1 bot_data = [create_NLP_dataframe(botfiles[i], 1) for i in range(len(botfiles))]
    

    <ipython-input-369-079eab89d369> in create_NLP_dataframe(list_of_users, botBool)
          1 def create_NLP_dataframe(list_of_users, botBool):
          2 
    ----> 3     tweets_df = pd.DataFrame.from_csv('data_NLP/bots/' + list_of_users[0] ,sep='\t')
          4     tweets_df.rename(index=str, columns={"full_text": "text"}, inplace=True)
          5 
    

    /anaconda3/lib/python3.5/site-packages/pandas/core/frame.py in from_csv(cls, path, header, sep, index_col, parse_dates, encoding, tupleize_cols, infer_datetime_format)
       1577                           parse_dates=parse_dates, index_col=index_col,
       1578                           encoding=encoding, tupleize_cols=tupleize_cols,
    -> 1579                           infer_datetime_format=infer_datetime_format)
       1580 
       1581     def to_sparse(self, fill_value=None, kind='block'):
    

    /anaconda3/lib/python3.5/site-packages/pandas/io/parsers.py in parser_f(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, escapechar, comment, encoding, dialect, tupleize_cols, error_bad_lines, warn_bad_lines, skipfooter, doublequote, delim_whitespace, low_memory, memory_map, float_precision)
        676                     skip_blank_lines=skip_blank_lines)
        677 
    --> 678         return _read(filepath_or_buffer, kwds)
        679 
        680     parser_f.__name__ = name
    

    /anaconda3/lib/python3.5/site-packages/pandas/io/parsers.py in _read(filepath_or_buffer, kwds)
        438 
        439     # Create the parser.
    --> 440     parser = TextFileReader(filepath_or_buffer, **kwds)
        441 
        442     if chunksize or iterator:
    

    /anaconda3/lib/python3.5/site-packages/pandas/io/parsers.py in __init__(self, f, engine, **kwds)
        785             self.options['has_index_names'] = kwds['has_index_names']
        786 
    --> 787         self._make_engine(self.engine)
        788 
        789     def close(self):
    

    /anaconda3/lib/python3.5/site-packages/pandas/io/parsers.py in _make_engine(self, engine)
       1012     def _make_engine(self, engine='c'):
       1013         if engine == 'c':
    -> 1014             self._engine = CParserWrapper(self.f, **self.options)
       1015         else:
       1016             if engine == 'python':
    

    /anaconda3/lib/python3.5/site-packages/pandas/io/parsers.py in __init__(self, src, **kwds)
       1706         kwds['usecols'] = self.usecols
       1707 
    -> 1708         self._reader = parsers.TextReader(src, **kwds)
       1709 
       1710         passed_names = self.names is None
    

    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader.__cinit__()
    

    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader._setup_parser_source()
    

    FileNotFoundError: File b'data_NLP/bots/7' does not exist




```python
#######################################################
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>created_at</th>
      <th>entities.user_mentions</th>
      <th>user_id</th>
      <th>num_hashtags</th>
      <th>num_upper</th>
      <th>all_emojis</th>
      <th>positive_emojis</th>
      <th>negative_emojis</th>
      <th>RT</th>
      <th>...</th>
      <th>VB_count</th>
      <th>VBD_count</th>
      <th>VBG_count</th>
      <th>VBN_count</th>
      <th>VBP_count</th>
      <th>VBZ_count</th>
      <th>WDT_count</th>
      <th>WP_count</th>
      <th>WP$_count</th>
      <th>WRB_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>thing bad technology writes managing director</td>
      <td>2018-11-08 16:39:01</td>
      <td>[{'screen_name': 'Campaignmag', 'id_str': '164...</td>
      <td>7087112</td>
      <td>0</td>
      <td>1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>174</th>
      <td>cma probe influencers fail declare payment end...</td>
      <td>2018-08-17 13:35:02</td>
      <td>[]</td>
      <td>7087112</td>
      <td>0</td>
      <td>1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>cma probe influencers fail declare payment end...</td>
      <td>2018-08-25 14:00:23</td>
      <td>[]</td>
      <td>7087112</td>
      <td>0</td>
      <td>1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>cma probe influencers fail declare payment end...</td>
      <td>2018-08-24 01:44:02</td>
      <td>[]</td>
      <td>7087112</td>
      <td>0</td>
      <td>1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>156</th>
      <td>cma probe influencers fail declare payment end...</td>
      <td>2018-08-19 16:02:07</td>
      <td>[]</td>
      <td>7087112</td>
      <td>0</td>
      <td>1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>





```python
# adjust spelling using TextBlob
# run once when data are ready- takes a long time! 
#tweets_df['text'].apply(lambda x: str(TextBlob(x).correct()))
```




```python
#tokenization
TextBlob(tweets_df['text'][0]).words

# convert word to base form
tweets_df['text'] = tweets_df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# build n-grams. That is, build some structures that often come together. For example, for Elon Musk, "Tesla 3."
# or for Trump, "build the wall," etc. 

TextBlob(tweets_df['text'][0]).ngrams(2)
```





    [WordList(['lining', '84']),
     WordList(['84', 'mannequin']),
     WordList(['mannequin', 'top']),
     WordList(['top', 'building']),
     WordList(['building', 'creating']),
     WordList(['creating', 'vineyard']),
     WordList(['vineyard', 'middle']),
     WordList(['middle', 'railway']),
     WordList(['railway', 'station']),
     WordList(['station', 'making']),
     WordList(['making', 'burger']),
     WordList(['burger', 'billboard']),
     WordList(['billboard', 'find']),
     WordList(['find', 'branded']),
     WordList(['branded', 'experience']),
     WordList(['experience', 'continue']),
     WordList(['continue', 'provoke']),
     WordList(['provoke', 'inspire']),
     WordList(['inspire', 'engage']),
     WordList(['engage', '2018'])]





```python
# TF-IDF explained https://www.analyticsvidhya.com/blog/2015/04/information-retrieval-system-explained/

# sample term frequency table
tf1 = (tweets_df['text'][0:10]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1.sort_values(['tf', 'words'], ascending=[0, 1]).head(10)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>words</th>
      <th>tf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>cma</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>declare</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>endorsement</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>fail</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>influencers</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>payment</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>probe</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>adblocking</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>behind</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>fall</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
display(tweets_df.head())
```




```python
# investigate which words appear frequently across texts -- these words don't give us much information (we,great,how,etc.)
for i,word in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(tweets_df.shape[0]/(len(tweets_df[tweets_df['text'].str.contains(word)])))
display(tf1.sort_values(['idf'],ascending=[1]).head(10))

tf1['tfidf'] = tf1['tf'] * tf1['idf']
display(tf1.sort_values(['tfidf'],ascending=[1]).head(10))

# can also do this with sklearn, see code below
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
#  stop_words= 'english',ngram_range=(1,1))
# train_vect = tfidf.fit_transform(tweets_df['text'])

# train_vect
```




```python
# build corpus
import tempfile
import logging
from gensim import corpora

corpus=[]
a=[]
for i in range(len(tweets_df['text'])):
        a=tweets_df['text'][i]
        corpus.append(a)
        
# look at first 5 lines        
corpus[0:5]

TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# remove common words and tokenize
# uncomment and download stopwords if nessisary
# nltk.download()
list1 = ['RT','rt']
stoplist = stopwords.words('english') + list(punctuation) + list1

# tokenize words 
texts = [[word for word in str(document).split()] for document in corpus]
```




```python
print(texts[1])
```




```python
# build dictionary of words, save it 
from collections import OrderedDict
from gensim import corpora, models, similarities
dictionary = corpora.Dictionary(texts)
dictionary.save(os.path.join(TEMP_FOLDER, 'twitter.dict'))

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'twitter.mm'), corpus)  # store to disk, for lda

# TfidfModel: multiplies a local component (term frequency) with a global component 
# (inverse document frequency), and normalizing the resulting documents to unit length
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors

total_topics = 5
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)
corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

#Show first n important word in the topics:
lda.show_topics(total_topics,5)

data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}

#made dataframe
df_lda = pd.DataFrame(data_lda)
print(df_lda.shape)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)
```




```python
%matplotlib inline

g=sns.clustermap(df_lda.corr(), center=0, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(12, 12))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()
```




```python
# look at results
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')
panel
```

