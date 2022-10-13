import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("/content/drive/MyDrive/Learning Data/spam.csv")

x = dataframe["EmailText"]
y = dataframe["Label"]

training, test = train_test_split(dataframe , test_size = 0.33 , random_state = 42)

train_x = training["EmailText"]
train_y = training["Label"]

test_x = test["EmailText"]
test_y = test["Label"]

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

clf_svm = svm.SVC(kernel = "linear")
clf_svm.fit(train_x_vectors , train_y)

clf_svm.predict(test_x_vectors[0])

print(clf_svm.score(test_x_vectors , test_y )*100)

test_date = [""]
