# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 12:47:09 2022

@author: VincentTse
"""

import bs4 as bs
import requests
import pandas as pd
import re
import concurrent.futures
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import spacy
#-------------------------------------------Web Scraping----------------------------------------------#

#Step 1-----Create a list of webpage links
pageLink = []   
searchFormat = 'https://datacenternews.asia/next?page={}'  #The target webpage
i = 1
while i <=300:
    pageLink.append(searchFormat.format(i))   #Get 300 pages on this webpage and store them into a list
    i += 1
    
    

#Step 2------Create a list of news links
newsLink = [] 
def function1 (pageLink):   #The function to get url of every news in one webpage. There are 300 webpages
    searchResp = requests.get(pageLink)    
    searchSoup = bs.BeautifulSoup(searchResp.text, 'lxml')  #Beautiful soup's function to get the HTML struture
    for x in searchSoup.findAll("a", {"class":"story-box"}):  
        newsLink.append(x.get('href'))     #A for loop to get the url for each news in each webpage
    
   
    
   
#Step 3------A function that create 4 lists, storing the title, date, content and url of every news     

#Prepare an empty list for append later
title = []     
content = []  
date = []
url = []  

def function2 (newsLink):  #The function to get title, date and content of each news. There are 7000+ news
    try: 
        linkResp = requests.get(newsLink)  #Use try except here to avoid visting error news webpage
    except:
        print(newsLink)     
    linkSoup = bs.BeautifulSoup(linkResp.text, 'lxml')   
    
    
    #The step to aviod visiting advertisement webpage during scraping
    if linkSoup.find('meta',{"property":"og:updated_time"}).get('content') == None: 
        pass   
    else:  #If there is valid content in the HTML, we are sure that it is now an advertisement webpage
        url.append(linkSoup.find('link',{"rel":"canonical"}).get('href'))  #Store the url for news to list
        date.append(linkSoup.find('meta',{"property":"og:updated_time"}).get('content')) #Store the date for news to list
        title.append(linkSoup.find('title').text) #Store the title for news to list
        string = []
        for i in linkSoup.findAll('p')[0:]:  
            string.append(i.text)  #Get all the paragraphs for the news 
    
        string = ' '.join(string) #Combine all paragraphs into one big long string
        content.append(string)   #Store the big string for news to list



#Step 4------Execute the functions with multi-threading
        
#Here because the web scraping process is slow, I use multi-threading to speed it up
maxThreads = 30  #Set the max threads
     
with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor: #Execute function1 
    executor.map(function1, pageLink) #Iterate through every webpage and store all news into a list
    
with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:   #Execute function2
    executor.map(function2, newsLink) #Iterate through every news webpage and store title, date, content into a list

       

#Step 5------Create a dataframe with 4 columns --> News title, News date, News content and News url
df = pd.DataFrame({'Title': title, 'Date':date, 'Content':content, 'Url':url})



#-------------------------------------------Data Cleaning----------------------------------------------#

#Clean up the date column and turn it to datetime format, so we can sort it from latest to oldest
df.Date = df.Date.str.replace(r"[T]",' ')
df.Date = df.Date.str.slice(0,-6)
df.Date = pd.to_datetime(df['Date'])
df = df.sort_values(['Date'], ascending=False)  #Sort the dataframe base on news date, from latest to oldest
df = df.reset_index(drop=True) #Reset index starting from 0


df.Content = df.Content.map(lambda x: re.sub('[,\.\'!?"\xa0]', '', x))
df.Content = df.Content.map(lambda x: x.lower())

#-------------------------------------------Text Processing----------------------------------------------#

#Step 1------Tokenize news content, replace important words
def token(i):
    for i in df.Content:  #Tokenize every single word into a list
         yield(gensim.utils.simple_preprocess(str(i), deacc=True)) 
matrix = list(token(df.Content))      


matrix = [[re.sub(r'\bcentre\b', 'center', token) for token in doc] for doc in matrix]
matrix = [[re.sub(r'\bcentres\b', 'center', token) for token in doc] for doc in matrix]
matrix = [[re.sub(r'\bcenters\b', 'center', token) for token in doc] for doc in matrix]

matrix = [[re.sub(r'\bacquire\b', 'acquisition', token) for token in doc] for doc in matrix]
matrix = [[re.sub(r'\bacquired\b', 'acquisition', token) for token in doc] for doc in matrix]
matrix = [[re.sub(r'\bacquires\b', 'acquisition', token) for token in doc] for doc in matrix]

matrix = [[re.sub(r'\bpartner\b', 'partnership', token) for token in doc] for doc in matrix]
matrix = [[re.sub(r'\bpartners\b', 'partnership', token) for token in doc] for doc in matrix]
matrix = [[re.sub(r'\bpartnered\b', 'partnership', token) for token in doc] for doc in matrix]

matrix = [[re.sub(r'\bmerge\b', 'merger', token) for token in doc] for doc in matrix]
matrix = [[re.sub(r'\bmerges\b', 'merger', token) for token in doc] for doc in matrix]
matrix = [[re.sub(r'\bmerged\b', 'merger', token) for token in doc] for doc in matrix]


#Step 2------Lemmatization noun only
def lemmatization(content, allowed_postags=['NOUN', 'PROPN']):  
    nlp = spacy.load("en_core_web_sm")  #Using spaCy library, loading English package here
    texts_out = []
    for i in content: #Operation for each news doc
        doc = nlp(" ".join(i)) #Join the words together first for lemma analysis
        #Get the lemma for each word, if the word is noun
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags]) 
    return texts_out
matrix = lemmatization(matrix, allowed_postags=['NOUN','PROPN'])
matrix = [[token.replace('datum', 'data') for token in doc] for doc in matrix]           


#Step 3------Create bigrams in text
def makeBigrams(content):  #Make bigrams for each news doc
    bigram = gensim.models.Phrases(matrix, min_count=5, threshold=100) # Higher threshold fewer phrases.
    return [bigram[doc] for doc in content]
matrix = makeBigrams(matrix)


#Step 4------Remove stopwords and useless words
def stopword(content):
    stopWords = stopwords.words('english') #Remove all the stopwords 
    stopWords.extend(['day', 'month', 'year', 'time', 'column', 'need', 'look', 'use', 'many', 'thing', 'lot', 'people', 'end'
                      , 'way', 'part', 'user', 'percent', 'cisco', 'window', 'lenovo', 'dell', 'australia', 'new_zealand', 'intel'
                      , 'share', 'quarter', 'management','city', 'gen','file', 'flash', 'ibm', 'sydney','singapore'])
    return [[i for i in simple_preprocess(str(doc)) 
             if i not in stopWords and len(i) >=3 ] for doc in content] #Also, remove words that length is smaller than 3
matrix = stopword(matrix) 


#Step 5------Create Corpus
matrixDict = corpora.Dictionary(matrix) 
corpus = [matrixDict.doc2bow(text) for text in matrix] #Corpus



#-------------------------------------------LDA Model----------------------------------------------#
#Step 1------Find the best parameter, which is the number of topic 
#The function generates serveral LDA models with different parameter, starting from 2 topics, 2 steps, and 40 topics max
def coherence(dictionary, corpus, texts, limit, start, step):
    values = []
    modelList = []
    for numTopics in range(start,limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, #Gensim library for LDA
                                           id2word=matrixDict,
                                           num_topics=numTopics,  #numTopics is the main parameter
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        modelList.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v') #Gensim library topic coherence
        values.append(coherencemodel.get_coherence())

    return modelList, values #'modelList' stores LDA models, 'values' stores the corresponding coherence value


modelList, values = coherence(dictionary=matrixDict, corpus=corpus, 
                                                        texts = matrix, start=6, limit=26 ,step=2)

for num, value in zip(range(6, 26, 2), values):
    print("Num Topics =", num, " has Coherence Value of", round(value, 4)) #Check highest coherence values for LDA models
    
    
    
    
#Step 2------Set up Lda models    
#Optimal choice of topics
lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=matrixDict,
                                           num_topics=8, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


#Step 3------Determine the topic for each news article
def topicAllocation(ldamodel=lda, corpus=corpus):
    # Init output
    df2 = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True) #Get the (topic, coherence values) tuple form the LDA object
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (num, value) in enumerate(row):
            if j == 0:  
                wp = ldamodel.show_topic(num)
                topicKeywords = ", ".join([word for word, prop in wp])
                df2 = df2.append(pd.Series([int(num), round(value,4), topicKeywords]), ignore_index=True)
            else:
                break
    df2.columns = ['Topic', 'Contribution', 'Keywords']

    # Add original dataframe to the end of the output
    df2 = pd.concat([df2, df], axis=1)
    return(df2)

#Step 4------Final output
dfFinal = topicAllocation(ldamodel=lda, corpus=corpus)

dfFinal.to_excel('output.xlsx')

