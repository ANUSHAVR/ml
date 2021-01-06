import pandas as pd
msg=pd.read_csv('naivetext.csv',names=['message','label'])
print('Total instances of the dataset:',msg.shape[0])
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
Y=msg.labelnum
print('The message and its label of first 5 instances are listed below')
X5, Y5 =X[0:5], msg.label[0:5]
for x, y in zip(X5,Y5):
    print(x, ',', y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X, Y)
print('Dataset is split into Training and Testing samples')
print ('the total number of Training Data :',xtrain.shape[0])
print ('the total number of Test Data :',xtest.shape[0])
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm=cv.transform(xtest)
print('Total features extracted using CountVectorizer:',xtrain_dtm.shape[1])
print('Features for first 5 training instances are listed below')
df=pd.DataFrame(xtrain_dtm.toarray(),columns=cv.get_feature_names())
print(df[0:5])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)
print('Classification results of testing samples are given below')
for doc,p in zip(xtest, predicted):
    pred = 'pos' if p==1 else 'neg'
    print('%s-> %s'%(doc,pred))
from sklearn import metrics
print('Accuracy metrics')
print('Accuracy of the classifier is',metrics.accuracy_score(ytest,predicted))
print('The value of Precision', metrics.precision_score(ytest,predicted))
print('The value of Recall', metrics.recall_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))

