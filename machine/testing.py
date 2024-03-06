import numpy as np
import nltk
import string
import random
import warnings

warnings.filterwarnings('ignore')

f = open("test.txt",'r',encoding="utf-8")
text = f.read()
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
text = text.lower()
sentenList = nltk.sent_tokenize(text)
wordList = nltk.word_tokenize(text)
# print(sentenList[:4])
# print(wordList[:4])
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

removpuncdict = dict((ord(punct),None)for punct in string.punctuation)

def Lemnormalisse(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(removpuncdict)))

greetInputs = ('hello','hey')
greetResponse = ('hey there','hi')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greetInputs:
            return random.choice(greetResponse)
        
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def reponse(userRes):
    robo = ''
    tfidvec = TfidfVectorizer(tokenizer= Lemnormalisse,stop_words='english')
    tfid = tfidvec.fit_transform(sentenList)
    vals = cosine_similarity(tfid[-1],tfid)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    reqtfid = flat[-2]
    if(reqtfid == 0):
        robo = robo + 'Didnt understand you'
        return robo
    else:
        robo = robo + sentenList[idx]
        return robo
    
flag = True
print("this is the bot")
while(flag):
    userRes = input()
    userRes = userRes.lower()
    if(userRes != 'bye'):
        if(userRes == 'thankyou'):
            flag = False
            print("you are welcome")
            
        else:
            if(greet(userRes) != None):
                print("bot: " + greet(userRes))
                
            else:
                sentenList.append(userRes)
                wordList = wordList + nltk.word_tokenize(userRes)
                finalw = list(set(wordList))
                print('bot: ',end = '')
                print(reponse(userRes))
                sentenList.remove(userRes)
                
    else:
        flag = False
        print('bot: Goodbye')


