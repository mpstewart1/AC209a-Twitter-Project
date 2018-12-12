---
title: AC209a Twitter Bot Detection Project
---

**Harvard University**

**Fall 2018 Semester**

**AC209a Project Group** : 15

**Group Members**: Matthew Stewart, Claire Stolz, Yiming Qin, and Tianning Zhao

## Project Introduction

Twitter is the battleground of modern politics, as is espoused daily by the aggressive tweets of political elites such as U.S. president Donald Trump. Having the ability to discern bots and legitimate humans on the social media platform will afford to opportunity to help curb fake news, and thus stop Trump from using it a catch-all excuse for political gain.

This project will look at analyzing Twitter data related and aims to develop a model capable of discriminating human accounts from bots using various machine learning and natural language processing techniques. Twitter data obtained from third party websites and from the native Twitter API `tweepy` will be used to train and test the models. Results for normal accounts can be checked using the 'http://botcheck.me' website, a well-known bot detection tool for Twitter. Proposed methods for the machine learning model include multiple logistic regression and artificial neural networks, with advanced procedures involving stacking/blending.

The major packages that are used during this project are the Twitter Python API known as **tweepy**, which allows us to extract real-time and historic tweets that can be analyzed. For the machine learning aspect of the project, the majority of the work will be completed using the natural language processing toolkit library known as **NLTK**, as well as the standard machine learning library for Python, **sklearn**. For neural networks we used **Keras** running **Tensorflow** on the back end. Advanced modeling techniques will be applied to the finalized dataset include stacking/blending, whereby multiple models are combined using a meta estimator to produce a superior model, much like the procedure of boosting. The blending package used is **mlens**.

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

The project also aims to provide insight into the following question:

**Can we identify the ratio of bots to legitimate users using a subset of tweets about Trump?**

The project has been split into several pages throughout this website, including:

- [Literature Review](https://mrdragonbear.github.io/AC209a-Twitter-Project/Literature_review.html)
- [Data Acquisition](https://mrdragonbear.github.io/AC209a-Twitter-Project/Twitter_data.html)
- [Exploratory Data Analysis](https://mrdragonbear.github.io/AC209a-Twitter-Project/EDA.html)
- [Model Development](https://mrdragonbear.github.io/AC209a-Twitter-Project/Final_Models.html)
- [Advanced Topics](https://mrdragonbear.github.io/AC209a-Twitter-Project/Advanced_Features.html)
- [Testing and Evaluation](https://mrdragonbear.github.io/AC209a-Twitter-Project/Testing_Evaluation.html)

![screenshot](/img/twitter.png){: .center-image }

## Summary

In summary, bot detection is not a trivial task and it is difficult to evaluate the performance of our bot detection algorithms due to the lack of certainty about real users scraped from the Twitter API. Even humans cannot achieve perfect accuracy in bot detection. That said, the models we developed over the course of this project were able to detect bots with a high accuracy. They even had comparable performance to well-developed models from industry trained on datasets ten times the size of ours and with more than 1000 features. Adding NLP-related features to our models significantly improved their accuracy on our test set. However, we were most effective at bot detection on an unseen data set using only user-based features. With more time, we would train our NLP models on a larger diversity of bots and users to allow for stronger fingerprinting of bots versus legitimate users.
