---
title: AC209a Twitter Bot Detection Project
---

**Harvard University**

**Fall 2018 Semester**

**AC209a Project Group** : 15

**Group Members**: Matthew Stewart, Claire Stolz, Yiming Qin, and Tianning Zhao

![screenshot](/img/bird_bot.png){: .center-image }

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

**How do bots and legitimate users differ in their discussion of political topics?**

The project has been split into several pages throughout this website, including:

- Data Acquisition
- Exploratory Data Analysis
- Model Development
- Advanced Topics
- Testing and Evaluation