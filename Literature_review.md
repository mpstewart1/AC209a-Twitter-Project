---
nav_include: 1
title: Literature Review
notebook: literature_Review.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

## Literature Review and Related Work

The following five journal articles were studied to review the standard analytical techniques used by researchers in the field of bot detection, in order to provide insight in how to implement our machine learning approach most effectively. In this section we will give a brief overview of the papers and the relevant knowledge extracted from them that aided our model development.

The five journal articles are:

1. [Deep Neural Networks for Bot Detection](https://arxiv.org/pdf/1802.04289.pdf)

2. [I Spot a Bot: Building a binary classifier to detect
bots on Twitter](https://pdfs.semanticscholar.org/e219/6b47133c2191d380098744c13ba77133e625.pdf)

3. [Detection of spam-posting accounts on Twitter](https://www.sciencedirect.com/science/article/pii/S0925231218308798)

4. [The Rise of Social Bots](https://arxiv.org/abs/1407.5225)

5. [Of Bots and Humans (on Twitter)](http://www-public.imtbs-tsp.eu/~farahbak/publications/Asonam_17.pdf)

### 1. Deep Neural Networks for Bot Detection

**Reference:**

*Sneha Kudugunta and Emilio Ferrara, "Deep Neural Networks for Bot Detection," Information Sciences, Volume 467, October 2018, Pages 312-322.* [Link here](https://arxiv.org/pdf/1802.04289.pdf).

**Overview:**

Kudugunta and Ferrara explore the use of a deep neural network for bot classification. The authors focus more on tweet-level detection than user-level detection, but they use the same datasets as our research, the Cresci and collaborators dataset (described in detail in our data discussion section). We emulate some of the account-level features they generate, including statuses count, followers count, friends count, favorites count, and default profile. We also utilize similar tweet-based features, including hashtag count, mention count, and retweet count. The authors find that Random Forest Classifiers and AdaBoost Classifiers produce the best results on account-level data. They are not able to replicate as high of an accuracy using tweet-level data. 

### 2. I Spot a Bot: Building a binary classifier to detect bots on Twitter

**Reference:**

*Jessica H. Wetstone and Sahil R. Nayyar, "I Spot a Bot: Building a binary classifier to detect bots on Twitter," CS 229 Final Project Report, December 2017.* [Link here](https://pdfs.semanticscholar.org/e219/6b47133c2191d380098744c13ba77133e625.pdf).

**Overview:** 

Wetstone and Nayyar use three different models to predict if a user is a bot. They implement a logistic regression model using 12 different features including follower count, favorites count, friend count, friend-to-follower ratio, hashtags and mentions per tweet, retweets per tweet, and unique geotags per tweet. They also utilize L2 regularization on their logistic model, with a constant of C=2.02 identified using cross-validation. Their second models was a gradient-boosted classifier, modeled of the Friedman "TreeBoost" algorithm. Finally, they use a Multi-Layer Neural Network, with two hidden layers and 3 and 4 nodes, respectively. They achieve 72.5%, 75.1%, and 77.7% accuracy on the test set for each model respectively. The authors mention in their discussion that their main source of error appears to come from an over-representation of bots in their training set, which results in an inaccurately high number of bots. While we also use a 50/50 split for bots and legitimate users, we pay particular attention to our rate of classification of bots in order to avoid this issue. 

### 3. Detection of spam-posting accounts on Twitter

**Reference:**

*Isa Inuwa-Dutse, Mark Liptrott, and Ioannis Korkontzelos, "Detection of spam-posting accounts on Twitter," Neurocomputing, Volume 315, November 2018, Pages 496-511.* [Link here](https://www.sciencedirect.com/science/article/pii/S0925231218308798). 

**Overview:**

The authors estimate that one in every 21 tweets is a spam message, with that rate increasing every year. The authors suggest the use of user, account, and pairwise engagement features to identify accounts. They reason that these features will be more persistent over time (the Twitter API only allows access to the last 3,200 tweets from a users). They attempt to use these features to accurately identify bots, and they find that rate of tweets is the best predictor of a bot-- typical bot accounts post over 12 tweets per day, and often in defined intervals. For their full model, they use account age, follower count, friends count, statuses count, digits in name, tweet length, user name length, screen name length, and a variety of other engagement measurements as features. Their best performing model was Gradient Boosting (98.8% area under ROC (AUC)) followed by Random Forest (98.6% AUC), and Extra Trees (98.6% AUC). We use all of these models in our analysis and are able to achieve similar accuracy scores for each. 

### 4. The Rise of Social Bots

**Reference** 

*Emilio Ferrara, Onur Varol, Clayton Davis, Filippo Menczer, and Alessandro Flammini, "The Rise of Social Bots," Communications of the ACM 59 (7), 96-104, 2016.* [Link here](https://arxiv.org/abs/1407.5225). 

**Overview:**

Ferrara et al. discuss the rise of social bots and the ever-more-important need for a Turing test to identify bots. They discuss the implications of social bots for the spread of misinformation and tampering with the social Web. They also discuss other uses for bots, such as sales and trading. The anthors continue on to discuss the increasing levels of sophistication of bots and the difficulties encountered when identifying them. Ferrara et al. report that some of the most important features for identifying bots are number of retweets, account age, number of tweets, number of replies, number of mentions, number of times retweeted, and username length. Specifically, they find that bots are more likely to have more retweets, have younger accounts, tweet less frequently, and have longer usernames. These are also features we include (and have found to be important) in our models.

### 5. Of Bots and Humans (on Twitter)

**Reference:** 

*Zafar Gilani, Reza Farahbakhsh, Gareth Tyson, Liang Wang, and Jon Crowcroft, "Of Bots and Humans (on Twitter)," Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2017, Pages 349-354.* [Link here](http://www-public.imtbs-tsp.eu/~farahbak/publications/Asonam_17.pdf).

**Overview:**

Gilani et al. discuss the behavior of bots and humans on Twitter. On the social analysis of bots and humans, they find that bots can successfully evade Twitter defenses, leading to increasing numbers of bots on Twitter. In fact, in a month, Twitter only suspended 38 of the 120 bots they implemented on Twitter. The authors also discuss the level of trust that users have in bots. In their analysis, the Gilani et al. examine the following: 1) Do bots generate more content than humans on Twitter? and 2) What do bot accounts Tweet? The authors find that bots accounted for 51.8% of all status updates in their dataset, and they find that bots are 2.2 times more likely to retweet than humans. They also find that human users are more popular and tend to have many more likes per tweet. Finally, they discuss the consumption habits of bots vs. humans. They find that bots are much less likely to favorite other content-- they tend to interact via retweeting rather than favoriting content. The authors also group their results by # of account followers (a metric for account popularity), an interesting direction but one we were not able to pursue due to Twitter API constraints. 
