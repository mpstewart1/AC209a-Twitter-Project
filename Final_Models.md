---
nav_include: 4
title: Model Development
notebook: Final_Models.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

### Important packages

1. `Tweepy` - **Twitter API** - [Documentation](http://docs.tweepy.org/en/v3.5.0/api.html#tweepy-api-twitter-api-wrapper)

2. `nltk` - **Natural language processing library** - [Documentation](http://www.nltk.org/howto/twitter.html)

3. `jsonpickle` - **Converts Python objects into JSON** - [Documentation](https://jsonpickle.github.io/)

4. `Pandas` - **Python Data Analysis Library** - [Documentation](https://pandas.pydata.org/)

5. `Matplotlib` - **Python data visualization library** - [Documentation](https://matplotlib.org/contents.html)

6. `Botometer` - **Bot checking library for Twitter** - [Documentation](https://market.mashape.com/OSoMe/botometer)

7. `Seaborn` - **Python data visualization library based on matplotlib** - [Documentation](https://seaborn.pydata.org/)

8. `scikit-learn` - **Python machine learning library** - [Documentation](https://scikit-learn.org/stable/documentation.html)

## Models

In this section we will go through each of the individual models worked through in the `Final_Models.ipynb` Jupyter notebook, assessing their performance on the training and test set, as well as the most important features for each model. The same 10 features were used for all models, which were taken from our dataframe that had been normalized such that regularization could be implemented, and all models with relevant hyperparameters had these tuned to attain the best accuracy achievable given the data and the assumptions of the model. The models discussed in this section are:

- Logistic Regression
- LDA/QDA
- Random Forest
- Boosting
- K Nearest Neighbors
- Feed Forward Artificial Neural Network
- Support Vector Machines
- Stacking (Meta Ensembling)
- Blended Ensemble

The overall results from these model and comparison between models is tackled in the discussion section. The overall aim for our models in this section is to obtain a high accuracy on the testing set whilst also minimizing the number of false positives, which would indicate that a human user is a bot. Minimizing false positives means that legitimate users are not unnecessary penalized if action is taken against bots in order to ameliorate the proliferance of bots on the social media platform.


### Logistic Regression

Two individual models were developed for logistic regression, first a linear logistic regression which only made use of the original 10 features from our dataframe. The second model added polynomial features of second order as well as interaction variables, which was applied using a data pipeline.

```python
logreg = LogisticRegression(C=100000,fit_intercept=True).fit(X_train_scaled,Y_train)
logreg_train = logreg.score(X_train_scaled, Y_train)
#accuracy_score(Y_train,logreg.predict(X_train_scaled), normalize=True)
print('Accuracy of logistic regression model on training set is {:.3f}'.format(logreg_train))
# Classification error on test set
#logreg_test = accuracy_score(logreg.predict(X_test_scaled), Y_test, normalize=True)
logreg_test = logreg.score(X_test_scaled, Y_test)
print('Accuracy of logistic regression model on the test set is {:.3f}'.format(logreg_test))
```

    Accuracy of logistic regression model on training set is: 0.776
    Accuracy of logistic regression model on the test set is: 0.775
    
As we can see for the basic logistic regression model, the model performs reasonably well on the dataset and is not overfitting. A confusion matrix will help us to assess the relative proportions of true positives and negatives and false positives and negatives when using this model.


```python
y_pred_logreg= logreg.predict(X_test_scaled)
df_confusion=pd.DataFrame(confusion_matrix(Y_test,y_pred_logreg))

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_confusion.index.name='Actual'
df_confusion.columns.name='Predicted'

def plot_confusion_matrix(df_confusion, title='Confusion matrix from logreg', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)
```

![png](Final_Models_files/Final_Models_14_0.png){: .center}

The confusion matrix for the basic logistic regression model shows us how well our model is performing in terms of testing accuracy as well as false positive rate. We see that our testing accuracy is around 80% and our false positive rating is between 0.2 and 0.3. Research papers estimate that around 10% of Twitter users are bots, which means that our model is not great as we are falsely predicting around 20% of the overall population. We should aim to reduce our false positive rate to below 10% to account for this.

So we can probably do better than the basic logistic regression model, let us first try by adding polynomial and interaction features and also cross-validation.


```python
# Logistic regression w/ quadratic + interaction terms + regularization
polynomial_logreg_estimator = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=True),
    LogisticRegressionCV(multi_class="ovr", penalty='l2', cv=5, max_iter=10000))
linearLogCVpoly = polynomial_logreg_estimator.fit(X_train_scaled, Y_train)
# Compare results
print('Polynomial-logistic accuracy: train={:.1%}, test={:.1%}'.format(
    linearLogCVpoly.score(X_train_scaled, Y_train), linearLogCVpoly.score(X_test_scaled, Y_test)))
linearLogCVpoly_train = linearLogCVpoly.score(X_train_scaled, Y_train)
linearLogCVpoly_test = linearLogCVpoly.score(X_test_scaled, Y_test)
```

    Polynomial-logistic accuracy: train=80.9%, test=80.0%
    
The accuracy of the polynomial model on the test set has been improved by several percent. We should once again check the confusion matrix and see if our false positive rating has also improved.


```python
y_pred_PolyL = linearLogCVpoly.predict(X_test_scaled)

df_confusion=pd.DataFrame(confusion_matrix(Y_test,y_pred_PolyL))

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_confusion.index.name='Actual'
df_confusion.columns.name='Predicted'

def plot_confusion_matrix(df_confusion, title='Confusion matrix from Poly-logistic', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)
```

![png](Final_Models_files/Final_Models_16_0.png){: .center}

Our confusion matrix gives approximately the same value as before. The logistic regression model does a pretty good job of separating bots from legimitate users with these features. However, it performs poorly in terms of predicting a large number of false positives. Let us leave logistic regression and move on to discriminant analysis.

### LDA and QDA Model

In this section we run LDA and QDA models to classify the users into either bots or legitimate users.

```python
lda = LinearDiscriminantAnalysis(store_covariance=True)
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
lda.fit(X_train_scaled, Y_train)
qda.fit(X_train_scaled, Y_train)
lda.predict(X_test_scaled)
qda.predict(X_test_scaled)

print('LDA accuracy train={:.1%}, test: {:.1%}'.format(
    lda.score(X_train_scaled, Y_train), lda.score(X_test_scaled, Y_test)))

lda_train = lda.score(X_train_scaled, Y_train)
lda_test = lda.score(X_test_scaled, Y_test)

print('QDA accuracy train={:.1%}, test: {:.1%}'.format(
    qda.score(X_train_scaled, Y_train), qda.score(X_test_scaled, Y_test)))

qda_train = qda.score(X_train_scaled, Y_train)
qda_test = qda.score(X_test_scaled, Y_test)
```

    LDA accuracy train=70.6%, test: 70.8%
    QDA accuracy train=70.8%, test: 71.1%
    
The LDA and QDA models run very quickly, which is one of their main advantages. However, as we see here, their performance on the test set is relatively poor in comparison to the logistic regression models. We see that LDA and QDA yield approximately the same values on the training and test set, which indicates that the assumption that the features are normally distributed is a reasonable assumption. Let us check the confusion matrix once again.

```python
y_pred_lda = lda.predict(X_test_scaled)

df_confusion=pd.DataFrame(confusion_matrix(Y_test,y_pred_lda))

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_confusion.index.name='Actual'
df_confusion.columns.name='Predicted'

def plot_confusion_matrix(df_confusion, title='Confusion matrix from LDA', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)
```

![png](Final_Models_files/Final_Models_20_0.png){: .center}

The confusion matrix for LDA shows that we are predicting a relatively low number of false positives, but we also have low values of true positives and false negatives. This is clearly not the best model to use. Let us see if QDA performs better.


```python
y_pred_qda = qda.predict(X_test_scaled)

df_confusion=pd.DataFrame(confusion_matrix(Y_test,y_pred_qda))

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_confusion.index.name='Actual'
df_confusion.columns.name='Predicted'

def plot_confusion_matrix(df_confusion, title='Confusion matrix from QDA', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)
```
![png](Final_Models_files/Final_Models_21_0.png){: .center}

QDA has a terrible performance in terms of false positives, which are nearly 40%! Clearly discriminant analysis is not the best model to use on this particular set of data. Now we can move on and try some bagging and boosting techniques. First we will try random forest.

### Random forest

The random forest model runs a number of iterations with bootstrapped samples, and develops models where the predictors are randomly selected at each node. This helps to introduce randomness in the model which can provide variance reduction. Hopefully random forest can perform better than our previous models.

```python
rf = RandomForestClassifier(n_estimators=50 , max_depth=15, max_features='auto')
rf.fit(X_train_scaled, Y_train)
rf_train =rf.score(X_train_scaled, Y_train)
rf_test =rf.score(X_test_scaled, Y_test)

print('RF accuracy train={:.1%}, test: {:.1%}'.format(rf_train,rf_test))
y_pred = rf.predict(X_test_scaled)

df_confusion=pd.DataFrame(confusion_matrix(Y_test,y_pred  ))

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_confusion.index.name='Actual'
df_confusion.columns.name='Predicted'
```

    RF accuracy train=99.4%, test: 91.4%
    
This is our highest test accuracy so far! This means that 9/10 of our predictions are correct, which is a big jump up from our previous models. But, we must still assess to see how high our false positive rates are using the confusion matrix. We will also check to see what the most important features are via variable importance.

```python
plt.figure(figsize=(5,5))
plt.title('Variable Importance from Random Forest')
plt.xlabel('Relative Importance')
pd.Series(rf.feature_importances_,index=list(X_train_scaled)).sort_values().plot(kind="barh")
```


    <matplotlib.axes._subplots.AxesSubplot at 0x1b41600b4e0>

![png](Final_Models_files/Final_Models_25_1.png){: .center}

```python
def plot_confusion_matrix(df_confusion, title='Confusion matrix from RF', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)
```

![png](Final_Models_files/Final_Models_26_0.png){: .center}

Our false positive rating for random forest is around 0.1, which is significantly better than our previous models. We are getting into the realm of acceptable accuracy with this model, but let's see if we can improve this further with a boosted model.

### AdaBoost and XGBoost

AdaBoost stands for adaptive boosting. The algorithm works by using decision tree classification, whereby the misclassified samples are used to adapt the model. This is done by successively adding models that are trained on the residuals of the misclassified samples to the original model, known as an additive model. Let's see how well AdaBoost performs.

XGBoost is a similar method to AdaBoost, and standards for extreme gradient boosting. This name comes from the optimization technique known as gradient descent that is implemented in the model. XGBoost is the most popular machine learning model utilized by Kagglers (a data science competition website) to obtain models with superior predictive capability.

```python
adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=800, learning_rate=0.01)
adaboost.fit(X_train_scaled, Y_train);
```

```python
y_pred = adaboost.predict(X_test_scaled)
pred_adaboost = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, pred_adaboost)

adaboost_train = adaboost.score(X_train_scaled, Y_train)
adaboost_test = adaboost.score(X_test_scaled, Y_test)

print("Adaboost Test Accuracy: %.2f%%" % (accuracy * 100.0))
df_confusion=pd.DataFrame(confusion_matrix(Y_test,y_pred  ))

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_confusion.index.name='Actual'
df_confusion.columns.name='Predicted'

def plot_confusion_matrix(df_confusion, title='Confusion matrix of adaboost', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)

```

    Adaboost Test Accuracy: 94.58%
    
![png](Final_Models_files/Final_Models_29_1.png){: .center}

This testing accuracy is even higher than the random forest model! This is clearly our best model so far, as we are only misclassifying around 5% of the test set. We should assess the variable importance and confusion matrix as we did for the random forest model.


```python
plt.figure(figsize=(5,5))
plt.title('Variable Importance from Adaboosting')
plt.xlabel('Relative Importance')
pd.Series(adaboost.feature_importances_,index=list(X_train_scaled)).sort_values().plot(kind="barh")
```

    <matplotlib.axes._subplots.AxesSubplot at 0x1b4161a66a0>

![png](Final_Models_files/Final_Models_30_1.png){: .center}


Interestingly, account age seems to be our most informative predictor, closely followed by number of friends and number of followers. The influence of account age could be because bots are more likely to be banned and thus new bots are continually created. However, we should be cautious when using this model as we are likely to overpredict that all new account are bots, which is clearly not true.

```python
import xgboost as xgb
from sklearn.metrics import confusion_matrix

xgb = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.01).fit(X_train_scaled, Y_train)
y_pred = xgb.predict(X_test_scaled)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)

xgb_train = xgb.score(X_train_scaled, Y_train)
xgb_test = xgb.score(X_test_scaled, Y_test)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
df_confusion=pd.DataFrame(confusion_matrix(Y_test,predictions))

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_confusion.index.name='Actual'
df_confusion.columns.name='Predicted'

def plot_confusion_matrix(df_confusion, title='Confusion matrix of xgboost', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)
```

    Accuracy: 92.22%
    
XGBoost also performs better than the random forest model, which is expected since we attained such a high result from our Adaboost model.

![png](Final_Models_files/Final_Models_31_1.png){: .center}

The confusion matrix for XGBoost shows that we have around 0.1 for false positives again, which is within an acceptable range. Whether this level of false positives is acceptable will be assessed in the model testing section.


```python
plt.figure(figsize=(5,5))
plt.title('Variable Importance from XGBoost')
plt.xlabel('Relative Importance')
pd.Series(xgb.feature_importances_,index=list(X_train_scaled)).sort_values().plot(kind="barh")
```

    <matplotlib.axes._subplots.AxesSubplot at 0x1b416299b70>

![png](Final_Models_files/Final_Models_32_1.png){: .center}


Once again, account age is by far the most important predictor for the XGBoost model.

### K Nearest Neighbors

K nearest neighbors is a non-parametric technique that looks at nearby points in order to classify a new sample. This is one of the most basic machine learning techniques but can often perform well due to its inherent lack of assumptions about the data. Let's see how well the model performs.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

scores_mean=[]
scores_std=[]

k_number=np.arange(1,40)

for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = k)
    score_mean= cross_val_score(knn,X_train_scaled,Y_train,cv=5).mean()
    score_std=cross_val_score(knn,X_train_scaled,Y_train,cv=5).std()
    scores_mean.append(score_mean)
```

```python
max_score_k=max(scores_mean)
best_k=scores_mean.index(max(scores_mean))+1
print('Best K=',best_k, 'with a max CV score of',max_score_k)

knn_best_k = KNeighborsClassifier(n_neighbors = best_k)
knn_best_k.fit(X_train_scaled,Y_train);

pred_best_k = knn_best_k.predict(X_test_scaled)

print('test accuracy',accuracy_score(Y_test, pred_best_k))

knn_best_k_train = knn_best_k.score(X_train_scaled, Y_train)
knn_best_k_test = knn_best_k.score(X_test_scaled, Y_test)
```

    Best K= 8 with a max CV score of 0.7114063374922827
    test accuracy 0.718296224588577
    

As should be expected from looking at the data in the EDA section, the KNN does not do as well as the other models. However, it is not the worst model we have seen and does indeed perform better than LDA and QDA.


```python
df_confusion=pd.DataFrame(confusion_matrix(Y_test,pred_best_k ))

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_confusion.index.name='Actual'
df_confusion.columns.name='Predicted'

def plot_confusion_matrix(df_confusion, title='Confusion matrix of KNN', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)
```

![png](Final_Models_files/Final_Models_35_0.png){: .center}

This confusion matrix tells us a similar story as our logistic regression plot.

After all of these different models, let us plot the accuracies of all of these models together and compare their performance. Further models involving stacking, blending, and K-means will be introduced in the advanced topics section and will be compared to these models.


```python
acc_scores=[accuracy_score(logreg.predict(X_test_scaled), Y_test),linearLogCVpoly.score(X_test_scaled, Y_test),lda.score(X_test_scaled, Y_test),qda.score(X_test_scaled, Y_test),accuracy_score(Y_test, pred_best_k),rf.score(X_test_scaled, Y_test),accuracy_score(Y_test, pred_adaboost),accuracy_score(Y_test, predictions)]

xx = [1,2,3,4,5,6,7,8]
index_name=['Linear Logistic','PolyLogistic', 'LDA','QDA','KNN','Random Forest','AdaBoosting','XGBoost']
plt.bar(xx, acc_scores)
plt.ylim(0.6,1)
plt.title('Accuracy score in each model ')
plt.ylabel('ACC Test')
plt.xticks(xx,index_name,rotation=90,fontsize = 14);
```

![png](Final_Models_files/Final_Models_36_0.png){: .center}


We see that our best model was AdaBoost, which achieved a 94.5% accuracy on the test set. Our worst models were KNN, LDA, and QDA. The models with the lowest level of false positives were AdaBoost, XGBoost, and random forest. These are the top three models that should be further pursued for the optimized bot detection model.

