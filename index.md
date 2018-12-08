---
title: AC209a Twitter Bot Detection Project
---

## Project Introduction

Twitter is the battleground of modern politics, as is espoused daily by the aggressive tweets of political elites such as U.S. president Donald Trump. Having the ability to discern bots and legitimate humans on the social media platform will afford to opportunity to help curb fake news, and thus stop Trump from using it a catch-all excuse for political gain.

This notebook will give an introduction to the planned AC209a term project, which will look at analyzing Twitter data related to the U.S. midterms and discriminating human accounts from bots using various natural language processing techniques. Topic modeling is proposed for preprocessing to find tweets which contain key words or relevant discussion topics, subsequently followed by feature engineering of these tweets. Suggested features include: follower count, account age, total number of tweets, retweet count, friends count, etc. Following this, we will implement several models to determine how well we can discriminate against bot accounts and humans. Well-known twitter bots will be added to the training and test set in order to check the accurcy of the model in determining which accounts are bots. Results for normal accounts can be checked using the 'http://botcheck.me' website. Proposed methods for the machine learning model include multiple logistic regression and artificial neural networks, with advanced procedures involving stacking/blending.

The major packages that will are used during this project are the Twitter Python API known as **tweepy**, which allows us to extract real-time and historic tweets that can be analyzed. For the machine learning aspect of the project, the majority of the work will be completed using the natural language processing toolkit library known as **NLTK**. For the multiple logistic regression model we will use the **sklearn** library and for neural networks we used **Keras** running **Tensorflow** on the back end. Advanced modeling techniques will be applied to the finalized dataset include stacking/blending, whereby multiple models are combined to produce a superior model, much like the procedure of boosting. The blending package that will be used is **mlens**.

The machine learning techniques used in this project are:

- Logistic Regression
- LDA/QDA
- Random Forest
- Boosting
- K Nearest Neighbors
- Feed Forward Artificial Neural Network
- Support Vector Machines
- Stacking (Meta Ensembling)
- Blended Ensemble

![screenshot](/img/bot.jpg){ width=50% }