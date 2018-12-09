---
nav_include: 1
title: Data Acquisition
notebook: Twitter_data.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

## Twitter API Login

The first thing that is needed is to gain access to the Twitter API, this involves setting up an account on the Twitter Development Platform, located [here](https://developer.twitter.com/content/developer-twitter/en.html). Once this is done, we have access to our access token and consumer key which allows us to connect to the API.

```python
# Log in to Twitter API
auth = tweepy.OAuthHandler('consumer_key', 'consumer_secret')
auth.set_access_token('access_token', 'access_token_secret')
api = tweepy.API(auth)
```

## Social

```python
# A dataset of (i) genuine, (ii) traditional, and (iii) social spambot Twitter accounts, annotated by CrowdFlower contributors. Released in CSV format.

# Genuine users
gu_df = pd.read_csv('./cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/users.csv', sep = ',')
gu_list = gu_df['id'].values.astype(int)

# Social spambots
ssbots1_df = pd.read_csv('./cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/users.csv', sep = ',')
ssbots1_list = ssbots1_df['id'].values.astype(int)
ssbots2_df = pd.read_csv('./cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/users.csv', sep = ',')
ssbots2_list = ssbots2_df['id'].values.astype(int)
ssbots3_df = pd.read_csv('./cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/users.csv', sep = ',')
ssbots3_list = ssbots3_df['id'].values.astype(int)

# traditional spambots
tsbots1_df = pd.read_csv('./cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/users.csv', sep = ',')
tsbots1_list = tsbots1_df['id'].values.astype(int)
tsbots2_df = pd.read_csv('./cresci-2017.csv/datasets_full.csv/traditional_spambots_2.csv/users.csv', sep = ',')
tsbots2_list = tsbots2_df['id'].values.astype(int)
tsbots3_df = pd.read_csv('./cresci-2017.csv/datasets_full.csv/traditional_spambots_3.csv/users.csv', sep = ',')
tsbots3_list = tsbots3_df['id'].values.astype(int)
tsbots4_df = pd.read_csv('./cresci-2017.csv/datasets_full.csv/traditional_spambots_4.csv/users.csv', sep = ',')
tsbots4_list = tsbots4_df['id'].values.astype(int)
```


```python
ssbots_list = list(ssbots1_list) + list(ssbots2_list) + list(ssbots3_list)
tsbots_list = list(tsbots1_list) + list(tsbots2_list) + list(tsbots3_list) + list(tsbots4_list)
```


```python
# Social Honeypot Dataset

# Legitimate user info
lu_df = pd.read_csv('./social_honeypot_icwsm_2011/legitimate_users.txt', sep = '\t', header = None)
lu_df.columns = ['UserID', 'CreatedAt', 'CollectedAt', 'NumerOfFollowings', 'NumberOfFollowers', 'NumberOfTweets', 'LengthOfScreenName', 'LengthOfDescriptionInUserProfile']
lu_tweets_df = pd.read_csv('./social_honeypot_icwsm_2011/legitimate_users_tweets.txt', sep = '\t', header = None)
lu_tweets_df.columns = ['UserID', 'TweetID', 'Tweet', 'CreatedAt']
lu_follow_df = pd.read_csv('./social_honeypot_icwsm_2011/legitimate_users_followings.txt', sep = '\t', header = None)
lu_follow_df.columns = ['UserID', 'SeriesOfNumberOfFollowings']

# Content polluters info
bots_df = pd.read_csv('./social_honeypot_icwsm_2011/content_polluters.txt', sep = '\t', header = None)
bots_df.columns = ['UserID', 'CreatedAt', 'CollectedAt', 'NumerOfFollowings', 'NumberOfFollowers', 'NumberOfTweets', 'LengthOfScreenName', 'LengthOfDescriptionInUserProfile']
bots_tweets_df = pd.read_csv('./social_honeypot_icwsm_2011/content_polluters_tweets.txt', sep = '\t', header = None)
bots_tweets_df.columns = ['UserID', 'TweetID', 'Tweet', 'CreatedAt']
bots_follow_df = pd.read_csv('./social_honeypot_icwsm_2011/content_polluters_followings.txt', sep = '\t', header = None)
bots_follow_df.columns = ['UserID', 'SeriesOfNumberOfFollowings']
```


```python
lu_list = lu_df['UserID'].values.astype(int)
bot_list = bots_df['UserID'].values.astype(int)
lu_df.head()
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
      <th>UserID</th>
      <th>CreatedAt</th>
      <th>CollectedAt</th>
      <th>NumerOfFollowings</th>
      <th>NumberOfFollowers</th>
      <th>NumberOfTweets</th>
      <th>LengthOfScreenName</th>
      <th>LengthOfDescriptionInUserProfile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>614</td>
      <td>2006-07-13 15:30:05</td>
      <td>2009-11-20 23:56:21</td>
      <td>510</td>
      <td>350</td>
      <td>3265</td>
      <td>10</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1038</td>
      <td>2006-07-15 16:12:15</td>
      <td>2009-11-16 05:12:11</td>
      <td>304</td>
      <td>443</td>
      <td>4405</td>
      <td>7</td>
      <td>156</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1437</td>
      <td>2006-07-16 12:29:24</td>
      <td>2009-11-16 16:25:12</td>
      <td>45</td>
      <td>73</td>
      <td>725</td>
      <td>6</td>
      <td>37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2615</td>
      <td>2006-07-19 23:23:55</td>
      <td>2009-11-27 18:34:36</td>
      <td>211</td>
      <td>230</td>
      <td>211</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3148</td>
      <td>2006-07-26 14:17:22</td>
      <td>2009-11-20 17:35:18</td>
      <td>7346</td>
      <td>7244</td>
      <td>11438</td>
      <td>8</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Given a name list and number of tweets needed to extract for each account
# Return a dictionary of dataframes
# Each dataframe contains info of one user
def API_scrap(name_list, count_num):
    fail_lst = []
    user_dfs = {}
    for name in name_list:
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
start = time.time()
gu_dfs, fail_lst = API_scrap(gu_list, 200)
end = time.time()
print('Elapsed time:', end-start)
```


```python
gu_full_df = create_df(gu_dfs, 'gu_dataframe')
```


```python
start = time.time()
ssbots_dfs, ssbots_fail_lst = API_scrap(ssbots_list, 200)
end = time.time()
print('Elapsed time:', end-start)
start = time.time()
tsbots_dfs, tsbots_fail_lst = API_scrap(tsbots_list, 200)
end = time.time()
print('Elapsed time:', end-start)

start = time.time()
sh_user_dfs, sh_fail_lst = API_scrap(lu_list, 200)
end = time.time()
print('Elapsed time:', end-start)
f = open("sh_user_dfs.pkl","wb")
pickle.dump(sh_user_dfs,f)
f.close()

start = time.time()
sh_bot_dfs, sh_bot_fail_lst = API_scrap(bot_list, 200)
end = time.time()
print('Elapsed time:', end-start)
f = open("sh_bot_dfs.pkl","wb")
pickle.dump(sh_user_dfs,f)
f.close()
```


```python
start = time.time()
sh_user_dfs, sh_fail_lst = API_scrap(lu_list, 10)
end = time.time()
print('Elapsed time:', end-start)
```

    Elapsed time: 6197.113664150238



```python
sh_user_full_df = create_df(sh_user_dfs, 'sh_user_dataframe')
```

    /anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2920: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    /anaconda3/lib/python3.6/site-packages/numpy/core/_methods.py:85: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)



```python
start = time.time()
sh_bot_dfs, sh_bot_fail_lst = API_scrap(bot_list, 10)
end = time.time()
print('Elapsed time:', end-start)
sh_bots_full_df = create_df(sh_bot_dfs, 'sh_bot_dataframe')
```

    Elapsed time: 3432.5196619033813



```python

```


```python
############ User features ############

# User ID
def user_id(df):
    try:
        return df['user.id_str'][0]
    except:
        return None
    
# Screen name length
def sname_len(df):
    try:
        return len(df['user.screen_name'][0])
    except:
        return None

# Number of digits in screen name
def sname_digits(df):
    try:
        return sum(c.isdigit() for c in df['user.screen_name'][0])
    except:
        return None
    
# User name length
def name_len(df):
    try: 
        return len(df['user.name'][0])
    except:
        return None

# Time offset (sec)
# df['user.time_zone'] or df['user.utc_offset']
# It is supposed to extract time info, but most of them were zeros

# Default profile
def def_profile(df):
    try:
        return int(df['user.default_profile'][0]*1)
    except:
        return None

# Default picture
def def_picture(df):
    try:
        return int(df['user.default_profile_image'][0]*1)
    except:
        return None

# Account age (in days)
def acc_age(df):
    try:
        d0 = datetime.strptime(df['user.created_at'][0],'%a %b %d %H:%M:%S %z %Y')
        d1 = datetime.now(timezone.utc)
        return (d1-d0).days
    except:
        return None

# Number of unique profile descriptions
def num_descrip(df):
    try:
        string = df['user.description'][0]
        return len(re.sub(r'\s', '', string).split(','))
    except:
        return None

# Number of friends
def friends(df):
    try:
        return df['user.friends_count'][0]
    except: 
        return None

# Number of followers
def followers(df):
    try:
        return df['user.followers_count'][0]
    except: 
        return None

# Number of favorites
def favorites(df):
    try:
        return df['user.favourites_count'][0]
    except:
        return None

# Number of tweets (including retweets, per hour and total)
def num_tweets(df):
    try:
        total = df['user.statuses_count'][0]
        per_hour = total/(acc_age(df)*24)
        return total, per_hour
    except:
        return None, None
```


```python
############ Timing features ############

def tweets_time(df):
    try:
        time_lst = []
        for i in range(len(df)-1):
            if df['retweeted'][i] == False:
                time_lst.append(df['created_at'][i])

        interval_lst = []
        for j in range(len(time_lst)-1):
            d1 = datetime.strptime(df['created_at'][j],'%a %b %d %H:%M:%S %z %Y')
            d2 = datetime.strptime(df['created_at'][j+1],'%a %b %d %H:%M:%S %z %Y')
            interval_lst.append((d2-d1).seconds)

        return np.array(interval_lst)
    except:
        return None
```


```python
# Content feature
def full_text(df):
    try:
        text_lst = []
        for i in range(len(df)):
            text_lst.append(df['full_text'][i])
        return text_lst
    
    except:
        return None
```


```python
def create_df(user_dfs, filename):
    columns_lst = ['User ID', 'Screen name length', 'Number of digits in screen name', 'User name length', 'Default profile (binary)','Default picture (binary)','Account age (days)', 'Number of unique profile descriptions','Number of friends','Number of followers','Number of favorites','Number of tweets per hour', 'Number of tweets total','timing_tweet']

    # Create user dataframe
    user_full_df = pd.DataFrame(columns = columns_lst)
    count = 0
    for name in user_dfs.keys():
        df = user_dfs[name]
        tweets_total, tweets_per_hour = num_tweets(df)
        data = [user_id(df), sname_len(df), sname_digits(df), name_len(df), def_profile(df), def_picture(df), acc_age(df), num_descrip(df), friends(df), followers(df), favorites(df), tweets_per_hour, tweets_total, np.mean(tweets_time(df))]
        user_full_df.loc[count] = data
        count += 1

    user_full_df = user_full_df.dropna()
    user_full_df.to_csv(filename+'.csv', encoding='utf-8', index=False)
    return user_full_df
```


```python
gu_full_df = create_df(gu_dfs, 'gu_dataframe')
```


```python
ssbots_full_df = create_df(ssbots_dfs, 'ssbots_dataframe')
```


```python
tsbots_full_df = create_df(tsbots_dfs, 'tsbots_dataframe')
```


```python
combined_bot_df = pd.concat([ssbots_full_df, tsbots_full_df], axis=0, sort=False)
```


```python
features = ['Screen name length', 'Number of digits in screen name', 'User name length', 'Account age (days)', 'Number of unique profile descriptions','Number of friends','Number of followers','Number of favorites','Number of tweets per hour', 'Number of tweets total','timing_tweet']
fig, axes = plt.subplots(len(features),1, figsize = (10,25))
for i in range(len(features)):
    sns.kdeplot(gu_full_df[features[i]], ax = axes[i], label = 'user')
    sns.kdeplot(combined_bot_df[features[i]], ax = axes[i], label = 'bot')
    #sns.kdeplot(tsbots_full_df[features[i]], ax = axes[i], label = 'bot2')
    axes[i].set_xlabel(features[i])
    axes[i].legend()

plt.tight_layout()
plt.savefig('DNA_features.png')
plt.show()
```


```python
sh_user_full_df = create_df(sh_user_dfs, 'sh_user_dataframe')
sh_bots_full_df = create_df(sh_bot_dfs, 'sh_user_dataframe')
```


```python
user_df1 = pd.read_csv('sh_user_dataframe.csv')
user_df2 = pd.read_csv('genuine_user_dataframe.csv')
user_df_final = user_df1.append(user_df2)
user_df_final.to_csv('user_df_final.csv', encoding='utf-8', index=False)

bot_df1 = pd.read_csv('ssbots_dataframe.csv')
bot_df2 = pd.read_csv('tsbots_dataframe.csv')
bot_df3 = pd.read_csv('sh_bot_dataframe.csv')
bot_df_final = bot_df1.append(bot_df2).append(bot_df3)
bot_df_final.to_csv('bot_df_final.csv', encoding='utf-8', index=False)
```

    4435
    1129



```python
len(user_df_final)
```




    5564




```python
# For 
columns_lst = ['User ID', 'Screen name length', 'Number of digits in screen name', 'User name length', 'Default profile (binary)','Default picture (binary)','Account age (days)', 'Number of unique profile descriptions','Number of friends','Number of followers','Number of favorites','Number of tweets per hour', 'Number of tweets total','timing_tweet']

# Create user dataframe
user_full_df = pd.DataFrame(columns = columns_lst)
count = 0
for name in user_dfs.keys():
    df = user_dfs[name]
    tweets_total, tweets_per_hour = num_tweets(df)
    data = [user_id(df), sname_len(df), sname_digits(df), name_len(df), def_profile(df), def_picture(df), acc_age(df), num_descrip(df), friends(df), followers(df), favorites(df), tweets_per_hour, tweets_total, np.mean(tweets_time(df))]
    user_full_df.loc[count] = data
    count += 1
    
user_full_df = user_full_df.dropna()

# Create bots dataframe
bots_full_df = pd.DataFrame(columns = columns_lst)
count = 0
for name in bot_dfs.keys():
    df = bot_dfs[name]
    tweets_total, tweets_per_hour = num_tweets(df)
    data = [user_id(df), sname_len(df), sname_digits(df), name_len(df), def_profile(df), def_picture(df), acc_age(df), num_descrip(df), friends(df), followers(df), favorites(df), tweets_per_hour, tweets_total, np.mean(tweets_time(df))]
    bots_full_df.loc[count] = data
    count += 1
    
bots_full_df = bots_full_df.dropna()

user_full_df.to_csv('user_dataframe.csv', encoding='utf-8', index=False)
bots_full_df.to_csv('bots_dataframe.csv', encoding='utf-8', index=False)
```


```python
columns_lst = ['User ID', 'Screen name length', 'Number of digits in screen name', 'User name length', 'Default profile (binary)','Default picture (binary)','Account age (days)', 'Number of unique profile descriptions','Number of friends','Number of followers','Number of favorites','Number of tweets per hour', 'Number of tweets total','timing_tweet']

# Create user dataframe
user_full_df = pd.DataFrame(columns = columns_lst)
count = 0
for name in user_dfs.keys():
    df = user_dfs[name]
    tweets_total, tweets_per_hour = num_tweets(df)
    data = [user_id(df), sname_len(df), sname_digits(df), name_len(df), def_profile(df), def_picture(df), acc_age(df), num_descrip(df), friends(df), followers(df), favorites(df), tweets_per_hour, tweets_total, np.mean(tweets_time(df))]
    user_full_df.loc[count] = data
    count += 1
    
user_full_df = user_full_df.dropna()

# Create bots dataframe
bots_full_df = pd.DataFrame(columns = columns_lst)
count = 0
for name in bot_dfs.keys():
    df = bot_dfs[name]
    tweets_total, tweets_per_hour = num_tweets(df)
    data = [user_id(df), sname_len(df), sname_digits(df), name_len(df), def_profile(df), def_picture(df), acc_age(df), num_descrip(df), friends(df), followers(df), favorites(df), tweets_per_hour, tweets_total, np.mean(tweets_time(df))]
    bots_full_df.loc[count] = data
    count += 1
    
bots_full_df = bots_full_df.dropna()

user_full_df.to_csv('user_dataframe.csv', encoding='utf-8', index=False)
bots_full_df.to_csv('bots_dataframe.csv', encoding='utf-8', index=False)
```

    /anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2920: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    /anaconda3/lib/python3.6/site-packages/numpy/core/_methods.py:85: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)



```python
# Example of the extracted user and timing features
user_full_df.head()
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
      <th>User ID</th>
      <th>Screen name length</th>
      <th>Number of digits in screen name</th>
      <th>User name length</th>
      <th>Default profile (binary)</th>
      <th>Default picture (binary)</th>
      <th>Account age (days)</th>
      <th>Number of unique profile descriptions</th>
      <th>Number of friends</th>
      <th>Number of followers</th>
      <th>Number of favorites</th>
      <th>Number of tweets per hour</th>
      <th>Number of tweets total</th>
      <th>timing_tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>614</td>
      <td>10</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>4519</td>
      <td>3</td>
      <td>2250</td>
      <td>1654</td>
      <td>5776</td>
      <td>0.141763</td>
      <td>15375</td>
      <td>74219.375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1038</td>
      <td>7</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>4517</td>
      <td>5</td>
      <td>1036</td>
      <td>1419</td>
      <td>4702</td>
      <td>0.313538</td>
      <td>33990</td>
      <td>79030.500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1437</td>
      <td>6</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>4516</td>
      <td>2</td>
      <td>210</td>
      <td>287</td>
      <td>405</td>
      <td>0.029451</td>
      <td>3192</td>
      <td>46867.500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2615</td>
      <td>7</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>4513</td>
      <td>4</td>
      <td>677</td>
      <td>758</td>
      <td>85</td>
      <td>0.007174</td>
      <td>777</td>
      <td>53885.750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3148</td>
      <td>8</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>4506</td>
      <td>3</td>
      <td>3834</td>
      <td>7956</td>
      <td>1629</td>
      <td>0.279276</td>
      <td>30202</td>
      <td>52754.125</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Feature statistics
user_df_final = pd.read_csv('user_df_final.csv')
bot_df_final = pd.read_csv('bot_df_final.csv')
features = ['Screen name length', 'Number of digits in screen name', 'User name length', 'Account age (days)', 'Number of unique profile descriptions','Number of friends','Number of followers','Number of favorites','Number of tweets per hour', 'Number of tweets total','timing_tweet']
fig, axes = plt.subplots(len(features),1, figsize = (10,25))
for i in range(len(features)):
    sns.kdeplot(user_df_final[features[i]], ax = axes[i], label = 'user')
    sns.kdeplot(bot_df_final[features[i]], ax = axes[i], label = 'bots')
    axes[i].set_xlabel(features[i])
    axes[i].legend()

plt.tight_layout()
plt.savefig('kdeplot_features.png')
plt.show()
```

    /anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](output_31_1.png)



```python
features = ['timing_tweet']
i = 0
fig = plt.gcf()
fig.set_size_inches(8,2)
sns.kdeplot(user_full_df[features[i]], label = 'user')
sns.kdeplot(bots_full_df[features[i]], label = 'bots')
    #axes[i].hist(user_full_df[features[i]], bins = 100, alpha = 0.5, density = True, label = 'user')
    #axes[i].hist(bots_full_df[features[i]], bins = 100, alpha = 0.5, density = True, label = 'bot')
    
plt.xlabel(features[i])
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig('kdeplot_features2.png')
plt.show()
```


![png](output_32_0.png)



```python
# Scatter matrix to see the correlation among extracted user and timing features
pd.scatter_matrix(user_full_df, figsize = (16,16))
plt.tight_layout()
plt.show()
```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: FutureWarning: pandas.scatter_matrix is deprecated, use pandas.plotting.scatter_matrix instead
      """Entry point for launching an IPython kernel.
    /anaconda3/lib/python3.6/site-packages/matplotlib/tight_layout.py:199: UserWarning: Tight layout not applied. tight_layout cannot make axes width small enough to accommodate all axes decorations
      warnings.warn('Tight layout not applied. '



![png](output_33_1.png)



```python
bot_df = pd.read_csv('bot_df_final.csv')
user_df = pd.read_csv('user_df_final.csv')
```


```python
fetched_tweets = api.search('trump', count = 10)
```


```python
fetched_tweets[0].user.id
```




    20118243




```python
import io
import time

def get_tweets(query, count):

    # empty list to store parsed tweets
    tweets = []
    target = io.open("mytweets.txt", 'w', encoding='utf-8')
    # call twitter api to fetch tweets
    q=str(query)
    
    for i in range(12):
        fetched_tweets = api.search(q, count = count)
        # parsing tweets one by one
        print(len(fetched_tweets))
        #print(fetched_tweets)
    
        for tweet in fetched_tweets:

            # empty dictionary to store required params of a tweet
            parsed_tweet = {}
            # saving text of tweet
            #print(tweet.id)
            #parsed_tweet['UserID'] = tweet.id
            tweets.append(tweet.user.id)
            if "http" not in tweet.text:
                line = re.sub("[^A-Za-z]", " ", tweet.text)
                target.write(line+"\n")
                
        time.sleep(10)

    return tweets

    # creating object of TwitterClient Class
    # calling function to get tweets
tweets = get_tweets(query ="Trump", count = 100)
```

    100
    100
    100
    100
    100
    100
    100
    100
    100
    100
    100
    100



```python
user_id_lst = list(set(tweets))
id_str_lst = [str(s) for s in user_id_lst]
start = time.time()
dfs_pred, fail_lst_pred = API_scrap(id_str_lst[0:1000], 10)
end = time.time()
print('Elapsed time:', end-start)
full_df_pred = create_df(dfs_pred, 'pred_dataframe')
```

    /anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:6211: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      sort=sort)


    Elapsed time: 589.1669261455536


    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:91: RuntimeWarning: divide by zero encountered in long_scalars



```python
mashape_key = "XXXX"
twitter_app_auth = {
    'consumer_key': 'XXXX',
    'consumer_secret': 'XXXX',
    'access_token': 'XXXX',
    'access_token_secret': 'XXXX',
  }
bom = botometer.Botometer(wait_on_ratelimit=True,
                          mashape_key=mashape_key,
                          **twitter_app_auth)

# Check a single account by screen name

result_lst = []
for user_id in user_id_lst[0:1000]:
    result = bom.check_account(user_id_lst[0])
    result.append(result)
    


# Check a single account by id
#result = bom.check_account(1548959833)

# Check a sequence of accounts
#accounts = ['@clayadavis', '@onurvarol', '@jabawack']
#for screen_name, result in bom.check_accounts_in(accounts):
    # Do stuff with `screen_name` and `result`
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-30-514013125648> in <module>
         12 # Check a single account by screen name
         13 
    ---> 14 result = bom.check_account(user_id_lst[0])
         15 
         16 # Check a single account by id


    /anaconda3/lib/python3.6/site-packages/botometer/__init__.py in check_account(self, user, full_user_object)
        124     def check_account(self, user, full_user_object=False):
        125         payload = self._get_twitter_data(user,
    --> 126                                          full_user_object=full_user_object)
        127         if not payload['timeline']:
        128             raise NoTimelineError(payload['user'])


    /anaconda3/lib/python3.6/site-packages/botometer/__init__.py in _get_twitter_data(self, user, full_user_object)
         76                     user,
         77                     include_rts=True,
    ---> 78                     count=200,
         79                     )
         80 


    /anaconda3/lib/python3.6/site-packages/tweepy/binder.py in _call(*args, **kwargs)
        248             return method
        249         else:
    --> 250             return method.execute()
        251 
        252     # Set pagination mode


    /anaconda3/lib/python3.6/site-packages/tweepy/binder.py in execute(self)
        162                                     if self.wait_on_rate_limit_notify:
        163                                         log.warning("Rate limit reached. Sleeping for: %d" % sleep_time)
    --> 164                                     time.sleep(sleep_time + 5)  # sleep for few extra sec
        165 
        166                 # if self.wait_on_rate_limit and self._reset_time is not None and \


    KeyboardInterrupt: 



```python
start = time.time()
result = bom.check_account(user_id_lst[0])
end = time.time()
result
```




    {'cap': {'english': 0.009922300094567935, 'universal': 0.007183846426565749},
     'categories': {'content': 0.4149558492738062,
      'friend': 0.20957193071779157,
      'network': 0.20385158449925947,
      'sentiment': 0.6700956285826207,
      'temporal': 0.2636844399869036,
      'user': 0.2533987749466703},
     'display_scores': {'content': 2.1,
      'english': 0.9,
      'friend': 1.0,
      'network': 1.0,
      'sentiment': 3.4,
      'temporal': 1.3,
      'universal': 0.7,
      'user': 1.3},
     'scores': {'english': 0.1807672693005155, 'universal': 0.13802618960216584},
     'user': {'id_str': '876476261220179968', 'screen_name': 'CKassube'}}




```python
end-start
```




    3.081282138824463




```python
id_str_lst = [str(s) for s in user_id_lst]
```
