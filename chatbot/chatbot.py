import nltk
import numpy as np
import string
f=open("data.txt","r",errors="ignore")
raw_doc=f.read()
raw_doc=raw_doc.lower()
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
sentance_tokenize=nltk.sent_tokenize(raw_doc)
word_tokenize=nltk.word_tokenize(raw_doc)
sentance_tokenize[:2]
lemmer=nltk.stem.WordNetLemmatizer()
def lemtokens(tokens):
  return[lemmer.lemmatize(token) for token in tokens]
remove_punt_dict=dict((ord(punct),None)for punct in string.punctuation)
def lemnormalize(text):
  return lemtokens(nltk.word_tokenize(text.lower().translate(remove_punt_dict)))
greet_inputs=["hi","hello","how are u"]
greeting_response=["hello there how can i help u","how you doing how can i assist u "]
def greeting(sentance):
  for word in sentance.split():
    if word.lower() in greet_inputs:
      return random.choice(greeting_response)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def response(user_response):
  robo1_response=""
  tfidvec=TfidfVectorizer(tokenizer=lemnormalize,stop_words="english")
  tfidf=tfidvec.fit_transform(sentance_tokenize)
  vals=cosine_similarity(tfidf[-1],tfidf)
  idx=vals.argsort()[0][-2]
  flat=vals.flatten()
  flat.sort()
  req_tfidf=flat[-2]
  if(req_tfidf==0):
      robo1_response=robo1_response+ "i am sorry i am unable to understand"
      return robo1_response
  else:
      robo1_response=robo1_response+sentance_tokenize[idx]
      return robo1_response
flag=True
print('hello i am bot')
while flag==True:
  user_response=input()
  user_response=user_response.lower()
  if(user_response!='bye'):
    if(user_response=='thank you'or user_response=='thanks'):
      flag=False
      print("bot:you are welcome")
    else:
      if(greeting(user_response)!=None):
        print("bot:"+greeting(user_response))
      else:
        sentance_tokenize.append(user_response)
        word_tokenize=word_tokenize+nltk.word_tokenize(user_response)
        final_words=list(set(word_tokenize))
        print("bot:",end="")
        print(response(user_response))
        sentance_tokenize.remove(user_response)
  else:
    flag=False
    print('bot:goodbye')
