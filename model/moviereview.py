import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use("ggplot")
import pickle 

import nltk
nltk.download('stopwords')

import nltk
nltk.download('punkt')



import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
# from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df=pd.read_csv("Sentiment_Analysis_on_Movie_Reviews_API\IMDB Dataset.csv")
df.head()

sns.countplot(x="sentiment", data=df)
plt.title("Sentimental Analysis")

df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 0, inplace=True)

def data_preprocessing(text):
  text=text.lower()
  text=re.sub("<br />", "", text)
  text=re.sub(r'https\S+|www\S+|http\S+', "", text, flags=re.MULTILINE)
  text=re.sub(r'https\S+|www\S+|http\S+', "", text, flags=re.MULTILINE)
  text=re.sub(r"\@w+|\#", "", text)
  text=re.sub(r"[^\w\s]", "", text)
  text_tokens=word_tokenize(text)
  filtered_text=[w for w in text_tokens if not w in stop_words]
  return " ".join(filtered_text)

df.review=df["review"].apply(data_preprocessing)

duplicated_count=df.duplicated().sum()
print("Number of duplicate entries: ", duplicated_count)

df=df.drop_duplicates("review")

stemmer= PorterStemmer()
def stemming(data):
  text=[stemmer.stem(word) for word in data]
  return data

df.review=df["review"].apply(lambda x: stemming(x))

pos_reviews=df[df.sentiment==1]
pos_reviews.head(5)

# text=" ".join([word for word in pos_reviews["review"]])
# plt.figure(figsize=(20,15), facecolor="None")
# wordcloud=WordCloud(max_words=500, width=1600, height=500).generate(text)
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.title("Most frequent words in positive reviews", fontsize=19)

from collections import Counter
count=Counter()
for text in pos_reviews["review"].values:
  for word in text.split():
    count[word]+=1
pos_words=pd.DataFrame(count.most_common(15))
pos_words.columns=["word", "count"]
pos_words.head()

px.bar(pos_words, x="count", y="word", title="Common words in Positive Reviews", color="word")

#Factorizing the Data
X= df["review"]
Y= df["sentiment"]

vect=TfidfVectorizer()
X=vect.fit_transform(df["review"])

x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, random_state=42)

print("Size of x_train: ", (x_train.shape))
print("Size of y_train: ", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))

#creating and saving a dataframe of train data
train_dict = {'review': x_train, "sentiment": y_train}
train_df = pd.DataFrame(train_dict)
train_df.to_csv('train.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings 
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score,mean_squared_error

logreg= LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred= logreg.predict(x_test)
logreg_acc= accuracy_score(logreg_pred, y_test)
print("Test Accuracy: {:.2f}%".format(logreg_acc*100))

print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))

mnb= MultinomialNB()
mnb.fit(x_train, y_train)
mnb_pred= mnb.predict(x_test)
mnb_acc= accuracy_score(mnb_pred, y_test)
print("Test Accuracy: {:.2f}%".format(mnb_acc*100))

print(confusion_matrix(y_test, mnb_pred))
print("\n")
print(classification_report(y_test, mnb_pred))

svc= LinearSVC()
svc.fit(x_train, y_train)
svc_pred= svc.predict(x_test)
svc_acc= accuracy_score(svc_pred, y_test)
print("Test Accuracy: {:.2f}%".format(svc_acc*100))

print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))

from sklearn.model_selection import GridSearchCV
param_grid={"C":[0.1, 1, 10, 100], "loss":["hinge", "squared_hinge"]}
grid= GridSearchCV(svc, param_grid, refit=True, verbose=3)
grid.fit(x_train, y_train)

print("Best Cross Validation Score: {:.2f}".format(grid.best_score_))
print("Best Parameters: ", grid.best_params_)

#Tuned SVC
svc2= LinearSVC(C=1, loss="hinge")
svc2.fit(x_train, y_train)
svc2_pred= svc.predict(x_test)
svc2_acc= accuracy_score(svc2_pred, y_test)
print("Test Accuracy: {:.2f}%".format(svc2_acc*100))

print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))

filename = 'best_model.sav'
pickle.dump(svc, open(filename, 'wb'))