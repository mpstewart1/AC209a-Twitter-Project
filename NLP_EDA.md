


```python
import tweepy
import nltk
import pandas as pd
import numpy as np
import sys
import jsonpickle
import os
import keras
import tensorflow
from textblob import TextBlob 
from plotly.offline import init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
import csv
import gensim
import logging
import tempfile
from nltk.corpus import stopwords
from textblob import Word
from string import punctuation
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import pyLDAvis.gensim
from bs4 import BeautifulSoup
import re
import emoji
import time

init_notebook_mode(connected=True) #do not miss this line

from gensim import corpora, models, similarities

import warnings
warnings.filterwarnings("ignore")
plotly.offline.init_notebook_mode()
```


    Using TensorFlow backend.
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-1-fde66de8f3bc> in <module>()
          9 import tensorflow
         10 from textblob import TextBlob
    ---> 11 from plotly.offline import init_notebook_mode, plot, iplot
         12 import plotly
         13 import plotly.graph_objs as go
    

    /anaconda3/lib/python3.5/site-packages/plotly/__init__.py in <module>()
         29 from __future__ import absolute_import
         30 
    ---> 31 from plotly import (plotly, dashboard_objs, graph_objs, grid_objs, tools,
         32                     utils, session, offline, colors, io)
         33 from plotly.version import __version__
    

    /anaconda3/lib/python3.5/site-packages/plotly/plotly/__init__.py in <module>()
          8 
          9 """
    ---> 10 from . plotly import (
         11     sign_in,
         12     update_plot_options,
    

    /anaconda3/lib/python3.5/site-packages/plotly/plotly/plotly.py in <module>()
         33 from plotly.plotly import chunked_requests
         34 
    ---> 35 from plotly.graph_objs import Scatter
         36 
         37 from plotly.grid_objs import Grid, Column
    

    /anaconda3/lib/python3.5/site-packages/plotly/graph_objs/__init__.py in <module>()
         26 from plotly.graph_objs import scatter3d
         27 from ._scatter import Scatter
    ---> 28 from plotly.graph_objs import scatter
         29 from ._sankey import Sankey
         30 from plotly.graph_objs import sankey
    

    /anaconda3/lib/python3.5/site-packages/plotly/graph_objs/scatter/__init__.py in <module>()
         11 from plotly.graph_objs.scatter import hoverlabel
         12 from ._error_y import ErrorY
    ---> 13 from ._error_x import ErrorX
    

    /anaconda3/lib/python3.5/importlib/_bootstrap.py in _find_and_load(name, import_)
    

    /anaconda3/lib/python3.5/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)
    

    /anaconda3/lib/python3.5/importlib/_bootstrap.py in _find_spec(name, path, target)
    

    /anaconda3/lib/python3.5/importlib/_bootstrap_external.py in find_spec(cls, fullname, path, target)
    

    /anaconda3/lib/python3.5/importlib/_bootstrap_external.py in _get_spec(cls, fullname, path, target)
    

    /anaconda3/lib/python3.5/importlib/_bootstrap_external.py in find_spec(self, fullname, target)
    

    /anaconda3/lib/python3.5/importlib/_bootstrap_external.py in _relax_case()
    

    KeyboardInterrupt: 




```python
# Consumer keys and access tokens, used for OAuth
consumer_key = '1K7KS692MTDzqWfHvvAPPSlqE'
consumer_secret = 'bRy95ycRGxry1EsSaQfnOVuZl31A5Sw6sn5xdXr2qcZkUmm0Aw'
access_token = '1053856790834753536-vLIZETTNJ9eFxsCtCNV4UCk0eOpj7g'
access_token_secret = 'gbJVvuKGYVJUAzXpWA2NhbaFezK5PBqJNfMhaObbXPDX7'

#auth = tweepy.OAuthHandler('pr0AH7Ot5sZmig4u3bA6j51ty', 'tNteF0tRlEjKJfkkWQaIv5myqT9oBqrIVOYPQJOMjBTJhn9SAF')
#auth.set_access_token('934846563825930241-yO5rosUB4x8eFMO0J7IXV1UZM0RzbgL', 'CbqfvlRonXo2JiIyxqCqeZynwkslNcDPmGFQ9KBEh8Mch')
 
#OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
# Creation of the actual interface, using authentication
api = tweepy.API(auth)

user = api.me()
 
print('Name: ' + user.name)
print('Location: ' + user.location)
print('Friends: ' + str(user.friends_count))

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
# load legitimate users file
legit_df = pd.DataFrame.from_csv('legitimate_users.txt',header=None,sep='\t')
legit_df.reset_index(inplace=True)
legit_list = list(legit_df[0])

# load bot users 
bot_df = pd.DataFrame.from_csv('bot_users.txt',header=None,sep='\t')
bot_df.reset_index(inplace=True)
bot_list = list(bot_df[0])

# temporarily only use first 1000 bots and legit users
legit_list = legit_list[0:1000]
bot_list = bot_list[0:1000]
```




```python
def get_all_tweets(screen_name, bots_list_bool):
    
    #Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
#     auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#     auth.set_access_token(access_token, access_token_secret)
#     api = tweepy.API(auth)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []
    
    try:
        
        try:
            # if user id is inputted 
            val = int(screen_name)

            #make initial request for most recent tweets (200 is the maximum allowed count)
            new_tweets = api.user_timeline(user_id = screen_name,count=200)#, tweet_mode = 'extended')

        except ValueError:
            # input was a string (screen name)
            #make initial request for most recent tweets (200 is the maximum allowed count)
            new_tweets = api.user_timeline(screen_name = screen_name,count=200)#,tweet_mode = 'extended')

        #save most recent tweets
        alltweets.extend(new_tweets)

        #save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        #keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            print("getting tweets before {}".format(oldest))

            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

            #save most recent tweets
            alltweets.extend(new_tweets)

            #update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1

            print("...{} tweets downloaded so far".format(len(alltweets)))

        screen_name = alltweets[0].user.screen_name
        user_id = alltweets[0].user.id_str
        
    
        #transform the tweepy tweets into a 2D array that will populate the csv	
        outtweets = [[screen_name, user_id, tweet.id_str, tweet.text,tweet.source] for tweet in alltweets
                    ]

        # potential other features
        #tweet.created_at, tweet.retweets, tweet.retweet_count, tweet.retweeted,tweet.lang,  tweet.geo,
        #              tweet.favorite, tweet.favorite_count, tweet.favorited,tweet.place,tweet.entities,
        
        #"created_at","retweets","retweet_count",
        #                     "retweeted","favorite", "favorite_count", "favorited", "entities","lang","place","geo",

        #write the csv

        # choose file path
        if bots_list_bool:
            filepath ='data_NLP/bots/{}_tweets.csv'
        else:
            filepath ='data_NLP/legit/{}_tweets.csv'

        with open(filepath.format(user_id), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["author","id","text","source"])
            writer.writerows(outtweets)

        pass
    except Exception as e: 
        print('Exception was:', e)
        print("Failed to load data for user {}".format(screen_name))
        pass
    
```




```python
# get first 10 bot users
for i in range(19,30):
    print("--- Fetching Legit User Number {}---".format(i))
    get_all_tweets(bot_list[i], True)
    #time.sleep(30)
```


    --- Fetching Legit User Number 19---
    getting tweets before 236127121695711231
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 2157321
    --- Fetching Legit User Number 20---
    getting tweets before 457959017667510271
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 2219081
    --- Fetching Legit User Number 21---
    getting tweets before 1048221407325253632
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 2243161
    --- Fetching Legit User Number 22---
    getting tweets before 1044143902645788671
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 2269491
    --- Fetching Legit User Number 23---
    getting tweets before 895774831987314688
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 2695911
    --- Fetching Legit User Number 24---
    getting tweets before 1067020154264010757
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 2884771
    --- Fetching Legit User Number 25---
    getting tweets before 1065242804492926976
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 3060631
    --- Fetching Legit User Number 26---
    getting tweets before 323291719829426177
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 3076591
    --- Fetching Legit User Number 27---
    getting tweets before 1061165868506365951
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 3136731
    --- Fetching Legit User Number 28---
    getting tweets before 814129310319726591
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 3291901
    --- Fetching Legit User Number 29---
    getting tweets before 529998427011026943
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 3881391
    



```python
# get first 10 legit users
for i in range(500,700):
    print("--- Fetching Legit User Number {}---".format(i))
    get_all_tweets(legit_list[i], False)
    #time.sleep(30)
    

```


    --- Fetching Legit User Number 500---
    getting tweets before 1061389470698659845
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7705112
    --- Fetching Legit User Number 501---
    getting tweets before 1065360401163083775
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7737912
    --- Fetching Legit User Number 502---
    getting tweets before 756232235892613119
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7760272
    --- Fetching Legit User Number 503---
    getting tweets before 950420201828360191
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7760472
    --- Fetching Legit User Number 504---
    getting tweets before 624406885484118015
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7777272
    --- Fetching Legit User Number 505---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7777392
    --- Fetching Legit User Number 506---
    getting tweets before 1008018937613778945
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7780092
    --- Fetching Legit User Number 507---
    Exception was: Not authorized.
    Failed to load data for user 7782002
    --- Fetching Legit User Number 508---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7801642
    --- Fetching Legit User Number 509---
    getting tweets before 26274659214
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7814852
    --- Fetching Legit User Number 510---
    getting tweets before 476566717963841538
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7842262
    --- Fetching Legit User Number 511---
    getting tweets before 1052060728578912255
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7870952
    --- Fetching Legit User Number 512---
    getting tweets before 692385776286437375
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7877092
    --- Fetching Legit User Number 513---
    getting tweets before 1067213061428260863
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7901732
    --- Fetching Legit User Number 514---
    getting tweets before 580817080628445184
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7919102
    --- Fetching Legit User Number 515---
    getting tweets before 87498187513933823
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7936262
    --- Fetching Legit User Number 516---
    getting tweets before 954728618076196866
    ...201 tweets downloaded so far
    getting tweets before 448010115074244607
    ...201 tweets downloaded so far
    --- Fetching Legit User Number 517---
    getting tweets before 905769194083545087
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7990062
    --- Fetching Legit User Number 518---
    getting tweets before 743056626886844419
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 7994952
    --- Fetching Legit User Number 519---
    getting tweets before 1055110223520821247
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8005012
    --- Fetching Legit User Number 520---
    getting tweets before 1007409515065536511
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8011432
    --- Fetching Legit User Number 521---
    getting tweets before 1063944919377620991
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8048482
    --- Fetching Legit User Number 522---
    getting tweets before 219040229418475519
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8053012
    --- Fetching Legit User Number 523---
    getting tweets before 764207944070201343
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8053532
    --- Fetching Legit User Number 524---
    getting tweets before 848731174352670720
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8068892
    --- Fetching Legit User Number 525---
    getting tweets before 1056186731173634047
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8073812
    --- Fetching Legit User Number 526---
    getting tweets before 1051358172622049279
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8074822
    --- Fetching Legit User Number 527---
    getting tweets before 1066280292099473407
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8077952
    --- Fetching Legit User Number 528---
    getting tweets before 554248196860686335
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8078672
    --- Fetching Legit User Number 529---
    getting tweets before 610869097312686079
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8095172
    --- Fetching Legit User Number 530---
    getting tweets before 1021081421035855871
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8105772
    --- Fetching Legit User Number 531---
    Exception was: Not authorized.
    Failed to load data for user 8123792
    --- Fetching Legit User Number 532---
    getting tweets before 880410177396244479
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8135492
    --- Fetching Legit User Number 533---
    getting tweets before 638776042560299007
    Exception was: Not authorized.
    Failed to load data for user 8141012
    --- Fetching Legit User Number 534---
    getting tweets before 100096906608394239
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8156282
    --- Fetching Legit User Number 535---
    getting tweets before 886565785543610368
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8156932
    --- Fetching Legit User Number 536---
    getting tweets before 905781979445989375
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8161082
    --- Fetching Legit User Number 537---
    getting tweets before 1065757590825644031
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8163442
    --- Fetching Legit User Number 538---
    getting tweets before 443771894354149375
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8167602
    --- Fetching Legit User Number 539---
    getting tweets before 1061419570466967551
    Exception was: Not authorized.
    Failed to load data for user 8175762
    --- Fetching Legit User Number 540---
    Exception was: Not authorized.
    Failed to load data for user 8180982
    --- Fetching Legit User Number 541---
    getting tweets before 1039171325086973951
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8182152
    --- Fetching Legit User Number 542---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8186762
    --- Fetching Legit User Number 543---
    getting tweets before 263370800323760127
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8188252
    --- Fetching Legit User Number 544---
    getting tweets before 1043762172134862847
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8203252
    --- Fetching Legit User Number 545---
    getting tweets before 813229198508752896
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8208492
    --- Fetching Legit User Number 546---
    getting tweets before 1059281633046478847
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8240772
    --- Fetching Legit User Number 547---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8248602
    --- Fetching Legit User Number 548---
    getting tweets before 850499743402872831
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8254252
    --- Fetching Legit User Number 549---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8257322
    --- Fetching Legit User Number 550---
    getting tweets before 436708824503042047
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8275692
    --- Fetching Legit User Number 551---
    getting tweets before 1061997737066094591
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8276062
    --- Fetching Legit User Number 552---
    getting tweets before 512504180582264831
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8282102
    --- Fetching Legit User Number 553---
    Exception was: Not authorized.
    Failed to load data for user 8294452
    --- Fetching Legit User Number 554---
    getting tweets before 1023024970224599040
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8295232
    --- Fetching Legit User Number 555---
    getting tweets before 1014540684412403711
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8309402
    --- Fetching Legit User Number 556---
    getting tweets before 1061370012475289599
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8318052
    --- Fetching Legit User Number 557---
    Exception was: Not authorized.
    Failed to load data for user 8328182
    --- Fetching Legit User Number 558---
    getting tweets before 193906754243936255
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8349012
    --- Fetching Legit User Number 559---
    getting tweets before 1062912698923372543
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8350872
    --- Fetching Legit User Number 560---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8359332
    --- Fetching Legit User Number 561---
    getting tweets before 217459660008071167
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8374772
    --- Fetching Legit User Number 562---
    getting tweets before 420152891102744575
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8375922
    --- Fetching Legit User Number 563---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8378752
    --- Fetching Legit User Number 564---
    getting tweets before 1046939397453533183
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8397122
    --- Fetching Legit User Number 565---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8427902
    --- Fetching Legit User Number 566---
    getting tweets before 1063483248750280704
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8446712
    --- Fetching Legit User Number 567---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8456492
    --- Fetching Legit User Number 568---
    getting tweets before 1063922694457053184
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8481222
    --- Fetching Legit User Number 569---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8486432
    --- Fetching Legit User Number 570---
    getting tweets before 1058435408072196095
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8490832
    --- Fetching Legit User Number 571---
    getting tweets before 511737508405587967
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8503472
    --- Fetching Legit User Number 572---
    getting tweets before 457522414977835007
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8539552
    --- Fetching Legit User Number 573---
    getting tweets before 1010116143955283967
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8544002
    --- Fetching Legit User Number 574---
    getting tweets before 1027919812050251775
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8548752
    --- Fetching Legit User Number 575---
    getting tweets before 955212955112693760
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8548952
    --- Fetching Legit User Number 576---
    getting tweets before 850992353590562815
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8552342
    --- Fetching Legit User Number 577---
    Exception was: Not authorized.
    Failed to load data for user 8554652
    --- Fetching Legit User Number 578---
    getting tweets before 582044385619107839
    ...198 tweets downloaded so far
    --- Fetching Legit User Number 579---
    getting tweets before 1062327115683631104
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8568192
    --- Fetching Legit User Number 580---
    getting tweets before 1064072710328770559
    ...201 tweets downloaded so far
    getting tweets before 108288039075319807
    ...201 tweets downloaded so far
    --- Fetching Legit User Number 581---
    getting tweets before 963683017070956543
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8608232
    --- Fetching Legit User Number 582---
    getting tweets before 523556749203480575
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8608612
    --- Fetching Legit User Number 583---
    getting tweets before 1056032658612871167
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8614192
    --- Fetching Legit User Number 584---
    getting tweets before 250953956954611711
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8619792
    --- Fetching Legit User Number 585---
    getting tweets before 1055081377752866817
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8627342
    --- Fetching Legit User Number 586---
    getting tweets before 1058743319197614080
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8639362
    --- Fetching Legit User Number 587---
    Exception was: Not authorized.
    Failed to load data for user 8660152
    --- Fetching Legit User Number 588---
    getting tweets before 515898192823848959
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8666172
    --- Fetching Legit User Number 589---
    getting tweets before 1040966360069808127
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8666612
    --- Fetching Legit User Number 590---
    Exception was: Not authorized.
    Failed to load data for user 8679982
    --- Fetching Legit User Number 591---
    getting tweets before 531433238464458751
    ...201 tweets downloaded so far
    getting tweets before 326205882893479935
    ...201 tweets downloaded so far
    --- Fetching Legit User Number 592---
    getting tweets before 1057698790453506047
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8703862
    --- Fetching Legit User Number 593---
    getting tweets before 1061633827867480064
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8705472
    --- Fetching Legit User Number 594---
    getting tweets before 709397774819729407
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8719572
    --- Fetching Legit User Number 595---
    getting tweets before 238155669809790975
    ...201 tweets downloaded so far
    getting tweets before 143350687462465536
    ...201 tweets downloaded so far
    --- Fetching Legit User Number 596---
    getting tweets before 1056685596570214399
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8732352
    --- Fetching Legit User Number 597---
    getting tweets before 915314674928902143
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8737532
    --- Fetching Legit User Number 598---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8740842
    --- Fetching Legit User Number 599---
    getting tweets before 1039315338419027967
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8747752
    --- Fetching Legit User Number 600---
    getting tweets before 1029741614179074047
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8763482
    --- Fetching Legit User Number 601---
    getting tweets before 1021859116636151807
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8767852
    --- Fetching Legit User Number 602---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8789632
    --- Fetching Legit User Number 603---
    getting tweets before 889604062412492802
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8821192
    --- Fetching Legit User Number 604---
    getting tweets before 364621785729212415
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8827302
    --- Fetching Legit User Number 605---
    getting tweets before 814107336159723521
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8835252
    --- Fetching Legit User Number 606---
    getting tweets before 56836951206133759
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8838312
    --- Fetching Legit User Number 607---
    getting tweets before 898664449917149183
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8858452
    --- Fetching Legit User Number 608---
    getting tweets before 926607713685164031
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8861742
    --- Fetching Legit User Number 609---
    getting tweets before 1064591826689044479
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8952542
    --- Fetching Legit User Number 610---
    getting tweets before 354370675890331649
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8974392
    --- Fetching Legit User Number 611---
    getting tweets before 537725886003302400
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 8980072
    --- Fetching Legit User Number 612---
    getting tweets before 913174178467115007
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9003212
    --- Fetching Legit User Number 613---
    getting tweets before 1052488525592096767
    ...200 tweets downloaded so far
    --- Fetching Legit User Number 614---
    Exception was: Not authorized.
    Failed to load data for user 9013332
    --- Fetching Legit User Number 615---
    getting tweets before 939195798855839744
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9014352
    --- Fetching Legit User Number 616---
    getting tweets before 727534268881084415
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9038492
    --- Fetching Legit User Number 617---
    getting tweets before 295393841006116863
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9056592
    --- Fetching Legit User Number 618---
    getting tweets before 76694769849483263
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9057432
    --- Fetching Legit User Number 619---
    getting tweets before 514087812128604159
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9058832
    --- Fetching Legit User Number 620---
    getting tweets before 863136917373804544
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9092762
    --- Fetching Legit User Number 621---
    getting tweets before 1039624319087923200
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9103722
    --- Fetching Legit User Number 622---
    getting tweets before 811159925359517695
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9103852
    --- Fetching Legit User Number 623---
    getting tweets before 1010628135572918272
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9128662
    --- Fetching Legit User Number 624---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9139022
    --- Fetching Legit User Number 625---
    getting tweets before 195813398242525183
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9140722
    --- Fetching Legit User Number 626---
    getting tweets before 1063202234589200384
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9161202
    --- Fetching Legit User Number 627---
    getting tweets before 509131789416554496
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9189862
    --- Fetching Legit User Number 628---
    getting tweets before 1021644051785961471
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9196572
    --- Fetching Legit User Number 629---
    getting tweets before 932353664429232127
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9198542
    --- Fetching Legit User Number 630---
    getting tweets before 1062759321950404607
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9207632
    --- Fetching Legit User Number 631---
    getting tweets before 1054357704536408074
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9209422
    --- Fetching Legit User Number 632---
    getting tweets before 1056937798006120447
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9228342
    --- Fetching Legit User Number 633---
    getting tweets before 136464751239761921
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9231442
    --- Fetching Legit User Number 634---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9244672
    --- Fetching Legit User Number 635---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9249352
    --- Fetching Legit User Number 636---
    Exception was: Not authorized.
    Failed to load data for user 9263172
    --- Fetching Legit User Number 637---
    getting tweets before 1065384626737139711
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9279222
    --- Fetching Legit User Number 638---
    getting tweets before 959546225941991424
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9280282
    --- Fetching Legit User Number 639---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9287612
    --- Fetching Legit User Number 640---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9292972
    --- Fetching Legit User Number 641---
    getting tweets before 1033553261867937791
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9298152
    --- Fetching Legit User Number 642---
    getting tweets before 1022813283869503487
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9302652
    --- Fetching Legit User Number 643---
    getting tweets before 293469760937660416
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9313112
    --- Fetching Legit User Number 644---
    getting tweets before 1038292115804041215
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9318752
    --- Fetching Legit User Number 645---
    getting tweets before 883415363974778879
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9328002
    --- Fetching Legit User Number 646---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9338842
    --- Fetching Legit User Number 647---
    getting tweets before 784493896906706944
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9346172
    --- Fetching Legit User Number 648---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9408252
    --- Fetching Legit User Number 649---
    getting tweets before 137222262397992959
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9431582
    --- Fetching Legit User Number 650---
    Exception was: Not authorized.
    Failed to load data for user 9452922
    --- Fetching Legit User Number 651---
    getting tweets before 1062093887907553282
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9460032
    --- Fetching Legit User Number 652---
    getting tweets before 1005151505178091519
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9484602
    --- Fetching Legit User Number 653---
    getting tweets before 1025962494378926079
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9499292
    --- Fetching Legit User Number 654---
    getting tweets before 19518148763
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9509312
    --- Fetching Legit User Number 655---
    Exception was: Not authorized.
    Failed to load data for user 9510792
    --- Fetching Legit User Number 656---
    Exception was: Not authorized.
    Failed to load data for user 9521532
    --- Fetching Legit User Number 657---
    getting tweets before 465298912824135679
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9527012
    --- Fetching Legit User Number 658---
    getting tweets before 1065008966588145663
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9532512
    --- Fetching Legit User Number 659---
    getting tweets before 970755614845603839
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9566042
    --- Fetching Legit User Number 660---
    getting tweets before 297364467602296831
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9569262
    --- Fetching Legit User Number 661---
    getting tweets before 27335068671
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9573852
    --- Fetching Legit User Number 662---
    getting tweets before 630877859901476863
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9583582
    --- Fetching Legit User Number 663---
    getting tweets before 1065562207923130369
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9587742
    --- Fetching Legit User Number 664---
    getting tweets before 1041846253951123457
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9594112
    --- Fetching Legit User Number 665---
    getting tweets before 1065174735418355711
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9608002
    --- Fetching Legit User Number 666---
    getting tweets before 672473584669999104
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9608132
    --- Fetching Legit User Number 667---
    getting tweets before 1065032902134640639
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9609372
    --- Fetching Legit User Number 668---
    getting tweets before 976161266447343615
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9626642
    --- Fetching Legit User Number 669---
    getting tweets before 1064858432258351103
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9634812
    --- Fetching Legit User Number 670---
    getting tweets before 18866939435
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9637032
    --- Fetching Legit User Number 671---
    getting tweets before 1054049181927534592
    Exception was: Not authorized.
    Failed to load data for user 9643542
    --- Fetching Legit User Number 672---
    getting tweets before 1055531050338828287
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9648662
    --- Fetching Legit User Number 673---
    getting tweets before 262192526889463808
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9650582
    --- Fetching Legit User Number 674---
    getting tweets before 1045437585479200767
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9655972
    --- Fetching Legit User Number 675---
    getting tweets before 165156403617599489
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9657882
    --- Fetching Legit User Number 676---
    getting tweets before 1061796076615684096
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9671202
    --- Fetching Legit User Number 677---
    getting tweets before 587957390315773951
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9677522
    --- Fetching Legit User Number 678---
    getting tweets before 978602809481408512
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9684422
    --- Fetching Legit User Number 679---
    getting tweets before 1032625371773054975
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9722452
    --- Fetching Legit User Number 680---
    getting tweets before 236398459429736447
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9772982
    --- Fetching Legit User Number 681---
    getting tweets before 1035791025422442495
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9779082
    --- Fetching Legit User Number 682---
    getting tweets before 1059834086326460415
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9782772
    --- Fetching Legit User Number 683---
    getting tweets before 10804275298
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9800782
    --- Fetching Legit User Number 684---
    getting tweets before 1052554423342260223
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9804112
    --- Fetching Legit User Number 685---
    Exception was: Not authorized.
    Failed to load data for user 9804712
    --- Fetching Legit User Number 686---
    getting tweets before 9576612756
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9815272
    --- Fetching Legit User Number 687---
    Exception was: Not authorized.
    Failed to load data for user 9820202
    --- Fetching Legit User Number 688---
    getting tweets before 1029170827713900543
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9841852
    --- Fetching Legit User Number 689---
    getting tweets before 644318883491508223
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9860352
    --- Fetching Legit User Number 690---
    getting tweets before 863578805239255039
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9865162
    --- Fetching Legit User Number 691---
    Exception was: Not authorized.
    Failed to load data for user 9878422
    --- Fetching Legit User Number 692---
    getting tweets before 1050543000831578111
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9880822
    --- Fetching Legit User Number 693---
    getting tweets before 681916132153016319
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9883162
    --- Fetching Legit User Number 694---
    getting tweets before 943866443044671493
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9895182
    --- Fetching Legit User Number 695---
    getting tweets before 136658709928607743
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9907912
    --- Fetching Legit User Number 696---
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9918772
    --- Fetching Legit User Number 697---
    getting tweets before 386631840637149183
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9981032
    --- Fetching Legit User Number 698---
    getting tweets before 714460335957213183
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9982862
    --- Fetching Legit User Number 699---
    getting tweets before 996090441580851202
    Exception was: [{'code': 34, 'message': 'Sorry, that page does not exist.'}]
    Failed to load data for user 9988572
    



```python
from os import listdir
from os.path import isfile, join
!ls
mypath_bots = 'data_NLP/bots/'
mypath_legit = 'data_NLP/legit/'
botfiles = [f for f in listdir(mypath_bots) if isfile(join(mypath_bots, f)) and not f=='.DS_Store']
legitfiles = [f for f in listdir(mypath_bots) if isfile(join(mypath_bots, f))and not f=='.DS_Store']
print(legitfiles)
```


    AM207_HW9_2018_Matthew-Stewart.ipynb [31mbot_users.txt[m[m
    BarackObama_tweets.csv               [34mcresci-2017.csv (1)[m[m
    LICENSE                              [34mdata_NLP[m[m
    NLP_EDA.ipynb                        elonmusk_tweets.csv
    README.md                            [31mlegitimate_users.txt[m[m
    _config.yml                          tweets.json
    ['10997_tweets.csv', '1599001_tweets.csv', '4567451_tweets.csv', '7967132_tweets.csv', '6301_tweets.csv', '11228722_tweets.csv', '10836_tweets.csv']
    



```python
get_all_tweets('elonmusk', False)
```


    getting tweets before 1057695407344541696
    ...400 tweets downloaded so far
    getting tweets before 1049363997617582080
    ...600 tweets downloaded so far
    getting tweets before 1038332059129761792
    ...800 tweets downloaded so far
    getting tweets before 1023749689701593087
    ...1000 tweets downloaded so far
    getting tweets before 1015854883990265855
    ...1200 tweets downloaded so far
    getting tweets before 1010428377797308415
    ...1400 tweets downloaded so far
    getting tweets before 1005875421999554559
    ...1600 tweets downloaded so far
    getting tweets before 1000444616536031231
    ...1800 tweets downloaded so far
    getting tweets before 996102919811350527
    ...2000 tweets downloaded so far
    getting tweets before 979895766880841732
    ...2200 tweets downloaded so far
    getting tweets before 960390661211021311
    ...2400 tweets downloaded so far
    getting tweets before 933576358793318400
    ...2600 tweets downloaded so far
    getting tweets before 901904930071609344
    ...2800 tweets downloaded so far
    getting tweets before 885626334503882751
    ...3000 tweets downloaded so far
    getting tweets before 872869542376030207
    ...3200 tweets downloaded so far
    getting tweets before 855161676705824767
    ...3224 tweets downloaded so far
    getting tweets before 848415356263702527
    ...3224 tweets downloaded so far
    



```python
!ls

tweets_df = pd.read_csv('elonmusk_tweets.csv')
# change to datetime
#tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], format='%Y-%m-%d %H:%M:%S')
full_tweets =tweets_df.copy()
#tweetsT = tweets_df['created_at']
tweets_df = tweets_df[['text']]
```


    AM207_HW9_2018_Matthew-Stewart.ipynb [31mbot_users.txt[m[m
    BarackObama_tweets.csv               [34mcresci-2017.csv (1)[m[m
    LICENSE                              [34mdata_NLP[m[m
    NLP_EDA.ipynb                        elonmusk_tweets.csv
    README.md                            [31mlegitimate_users.txt[m[m
    _config.yml                          tweets.json




```python
# get features of tweets (pre-cleaning data)

# number of hashtags 
tweets_df['num_hashtags'] = tweets_df['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

# number of all-caps words 
tweets_df['all_upper'] = tweets_df['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
```




```python
tweets_df['text'].values[4]
```





    '@margrethmpossi Varies per person, but about 80 sustained, peaking above 100 at times. Pain level increases exponentially above 80.'





```python
print(len(tweets_df['text']))
tweets_df['text'].values[3223]
```


    3224
    




    "@verge It won't matter"





```python
# deal with emojis

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

def getTweetEmoticons(tweet):
    print(tweet)
    regexp = {"SPACES": r"\s+"}
    emoji = list()
    for tok in re.split(ParseTweet.regexp["SPACES"], tweet.strip()):
        if tok in Emoticons.POSITIVE:
            emoji.append(tok)
        if tok in Emoticons.NEGATIVE:
            emoji.append(tok)
    return emoji

# # all emojis in a text 

def extract_emojis(str):
    return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)
tweets_df['emojis'] = [extract_emojis(tweet) for tweet in tweets_df['text'].values[i] 
                       for i in range(len(tweets_df['text']))]
#tweets_df['emojis'] =[getTweetEmoticons(text) for text in (tweets_df['text'].values[i]) 
#                      for i in range(len(tweets_df['text']))]  


```



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-135-7ef3ce98db71> in <module>()
         41 def extract_emojis(str):
         42     return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)
    ---> 43 tweets_df['emojis'] = [extract_emojis(tweet) for tweet in tweets_df['text'].values[i] 
         44                        for i in range(len(tweets_df['text']))]
         45 #tweets_df['emojis'] =[getTweetEmoticons(text) for text in (tweets_df['text'].values[i])
    

    /anaconda3/lib/python3.5/site-packages/pandas/core/frame.py in __setitem__(self, key, value)
       3117         else:
       3118             # set column
    -> 3119             self._set_item(key, value)
       3120 
       3121     def _setitem_slice(self, key, value):
    

    /anaconda3/lib/python3.5/site-packages/pandas/core/frame.py in _set_item(self, key, value)
       3192 
       3193         self._ensure_valid_index(value)
    -> 3194         value = self._sanitize_column(key, value)
       3195         NDFrame._set_item(self, key, value)
       3196 
    

    /anaconda3/lib/python3.5/site-packages/pandas/core/frame.py in _sanitize_column(self, key, value, broadcast)
       3389 
       3390             # turn me into an ndarray
    -> 3391             value = _sanitize_index(value, self.index, copy=False)
       3392             if not isinstance(value, (np.ndarray, Index)):
       3393                 if isinstance(value, list) and len(value) > 0:
    

    /anaconda3/lib/python3.5/site-packages/pandas/core/series.py in _sanitize_index(data, index, copy)
       3999 
       4000     if len(data) != len(index):
    -> 4001         raise ValueError('Length of values does not match length of ' 'index')
       4002 
       4003     if isinstance(data, ABCIndexClass) and not copy:
    

    ValueError: Length of values does not match length of index




```python
test_text = tweets_df.text[:5]
for i in test_text:
    print(i)
```


    1000 liters of pure water costs 1
    For those worried about running out of fresh water it may help to know that desalination only costs 01 cents per 
    Ah yes fair point
    Used to live in Silicon Valley now I live in Silicone Valley
    Not yet
    



```python
# clean tweets 
tweets_df['text'] = [re.sub(r'http[A-Za-z0-9:/.]+','',str(tweets_df['text'][i])) for i in range(len(tweets_df['text']))]
removeHTML_text = [BeautifulSoup(tweets_df.text[i], 'lxml').get_text() for i in range(len(tweets_df.text))]
tweets_df.text = removeHTML_text
tweets_df['text'] = [re.sub(r'@[A-Za-z0-9]+','',str(tweets_df['text'][i])) for i in range(len(tweets_df['text']))]


weird_characters_regex = re.compile(r"[^\w\d ]") 
tweets_df.text = tweets_df.text.str.replace(weird_characters_regex, "")


# look at results 
display(tweets_df.text[0])
display(tweets_df.head(20))
```



    '1000 liters of pure water costs 1'



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
      <th>num_hashtags</th>
      <th>all_upper</th>
      <th>emojis</th>
      <th>word_count</th>
      <th>char_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000 liters of pure water costs 1</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>7</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>For those worried about running out of fresh w...</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>22</td>
      <td>113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ah yes fair point</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>4</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Used to live in Silicon Valley now I live in S...</td>
      <td>0</td>
      <td>1</td>
      <td></td>
      <td>12</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Not yet</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>trouble</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Earth is</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Earths crust is only 1 of Earth mass so techni...</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Earth should be called Water Our surface is 71...</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>19</td>
      <td>94</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Yeah good idea to offer that as a setting</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>9</td>
      <td>41</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Contour remains approx same but fundamental ma...</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>13</td>
      <td>90</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RT _Alex This is real How a space ship leaves ...</td>
      <td>0</td>
      <td>2</td>
      <td></td>
      <td>25</td>
      <td>127</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RT    And heres a nifty spec sheet too</td>
      <td>0</td>
      <td>1</td>
      <td></td>
      <td>14</td>
      <td>41</td>
    </tr>
    <tr>
      <th>13</th>
      <td>If you dont want a Tesla heres a list of all e...</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>19</td>
      <td>81</td>
    </tr>
    <tr>
      <th>14</th>
      <td>RT  Trump administrations first report on clim...</td>
      <td>0</td>
      <td>2</td>
      <td></td>
      <td>19</td>
      <td>118</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Signing off for a while</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>5</td>
      <td>23</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Happy Thanksgiving</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>2</td>
      <td>18</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Starships were meant to fly</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>7</td>
      <td>29</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Starship Tooters</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>2</td>
      <td>16</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Good one</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>2</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python

```





    []





```python
# get word coult, char count
tweets_df['word_count'] = tweets_df['text'].apply(lambda x: len(str(x).split(" ")))
tweets_df['char_count'] = tweets_df['text'].str.len() ## this also includes spaces
display(tweets_df[['text','char_count','word_count']].head())

# check data types 
display(tweets_df.dtypes)
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
      <th>char_count</th>
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000 liters of pure water costs 1</td>
      <td>33</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>For those worried about running out of fresh w...</td>
      <td>113</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ah yes fair point</td>
      <td>17</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Used to live in Silicon Valley now I live in S...</td>
      <td>60</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Not yet</td>
      <td>7</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



    text            object
    num_hashtags     int64
    all_upper        int64
    emojis          object
    word_count       int64
    char_count       int64
    dtype: object




```python
# get average word length
def avg_word(sentence):
    words = sentence.split()
    if len(words) == 0:
        return 0
    return (sum(len(word) for word in words)/len(words))

tweets_df['avg_word'] = tweets_df['text'].apply(lambda x: avg_word(x))
tweets_df[['text','avg_word']].head()
```




```python
# all lowercase 
tweets_df['text'] = tweets_df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
display(tweets_df['text'].head())
```




```python
# remove stopwords and punctuation
retweet = ['RT','rt']
stoplist = stopwords.words('english') + list(punctuation) + retweet
#stop = stopwords.words('english')
tweets_df['text'] = tweets_df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stoplist))
tweets_df['text'].head()
```




```python
# add sentiment feature
tweets_df['sentiment'] = tweets_df['text'].apply(lambda x: TextBlob(x).sentiment[0])
tweets_df['polarity'] = tweets_df['text'].apply(lambda x: TextBlob(x).sentiment[1])
tweets_df = tweets_df.sort_values(['sentiment'])
```




```python
display(tweets_df[['text','sentiment','polarity']].head())
```




```python
# add list of nouns
tweets_df['nouns'] = [TextBlob(tweets_df.text[i]).noun_phrases for i in range(len(tweets_df.text))]
```




```python
tweets_df['classify'] = [TextBlob(tweets_df.text[i]).noun_phrases for i in range(len(tweets_df.text))]
```




```python
# look at 10 most frequent words
freq = pd.Series(' '.join(tweets_df['text']).split()).value_counts()[:10]
print(freq)

# look at 10 least frequent words
freq = pd.Series(' '.join(tweets_df['text']).split()).value_counts()[-20:]
freq
```




```python
# adjust spelling using TextBlob
# run once when data are ready- takes a long time! 
#tweets_df['text'].apply(lambda x: str(TextBlob(x).correct()))
```




```python
#tokenization
TextBlob(tweets_df['text'][0]).words
```




```python
# convert word to base form
tweets_df['text'] = tweets_df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
```




```python
# build n-grams. That is, build some structures that often come together. For example, for Elon Musk, "Tesla 3."
# or for Trump, "build the wall," etc. 

TextBlob(tweets_df['text'][0]).ngrams(2)
```




```python
# TF-IDF explained https://www.analyticsvidhya.com/blog/2015/04/information-retrieval-system-explained/

# sample term frequency table
tf1 = (tweets_df['text'][0:10]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1.sort_values(['tf', 'words'], ascending=[0, 1]).head(10)
```




```python
# investigate which words appear frequently across texts -- these words don't give us much information (we,great,how,etc.)
for i,word in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(tweets_df.shape[0]/(len(tweets_df[tweets_df['text'].str.contains(word)])))
display(tf1.sort_values(['idf'],ascending=[1]).head(10))
```




```python
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
corpus=[]
a=[]
for i in range(len(tweets_df['text'])):
        a=tweets_df['text'][i]
        corpus.append(a)
        
# look at first 5 lines        
corpus[0:5]

TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

from gensim import corpora
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
pyLDAvis.enable_notebook()
panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')
panel
```


Possible features to explore:

-Topic

-Diversity of words

-Diversity of topics

-Word density (words/tweet)

-Top words



```python
# plot tweet activity

trace = go.Histogram(
    x=tweetsT,
    marker=dict(
        color='blue'
    ),
    opacity=0.75
)

layout = go.Layout(
    title='Tweet Activity Over Years',
    height=450,
    width=1200,
    xaxis=dict(
        title='Month and year'
    ),
    yaxis=dict(
        title='Tweet Quantity'
    ),
    bargap=0.2,
)

data = [trace]

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
```

