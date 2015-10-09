# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 00:38:48 2015
@authors: Rohan Kulkarni , Aishwarya Rajesh

"""

import urllib2
import urllib
import base64
import json
import collections
import math
import numpy as np
from stop_words import get_stop_words
import nltk
import re


def bing_api(bingUrl,accountKey):
    """
    Query the Bing API and retrieve the results in the variable 'content'
    """
    accountKeyEnc = base64.b64encode(accountKey + ':' + accountKey)
    headers = {'Authorization': 'Basic ' + accountKeyEnc}
    req = urllib2.Request(bingUrl, headers = headers)
    response = urllib2.urlopen(req)
    content = response.read()
    return content

def stripPunctuation(input_string):
    """
    This function eliminates the common punctuations in the results received     
    """
    punctuation_tuple=('@','#','!','.',':',',','-','\\','/','|','(',')','&','[',']','"')
    stripped_str = "".join(c for c in input_string if c not in punctuation_tuple).encode('utf-8')     
    return str.split(str(stripped_str))
    
def hasNumber(key):
    """
    Checks if a word has a numerical entity 
    """
    number = re.search(r'\d+', key)
    return number
    
def updateDictionary(word_dictionary,qdata):
    """
    Creates a dictionary object out of the document descriptions received
    from Bing.    
    """
    stop_words = get_stop_words('english')
    for each in qdata:
        stripped_desc_list=stripPunctuation(each['description'])
        stripped_title_list=stripPunctuation(each['title'])
        data=stripped_desc_list+stripped_title_list
        for i in data:
            if i in word_dictionary:
                word_dictionary[i]+=1
            else:
                word_dictionary[i]=1
    for key in word_dictionary.keys():
        if key.decode('utf-8') in stop_words or key.isdigit() or hasNumber(key)!=None:
            del word_dictionary[key]
    ordered_dict = collections.OrderedDict(sorted(word_dictionary.items()))
    return ordered_dict

def findDocFrequency(word,all_docs):
    """
    Calculate Document Frequencies(#times a word appears in all documents) for all the words
    """
    df=0    
    for doc in all_docs:
        stripped_desc_list=stripPunctuation(doc['description'])
        stripped_title_list=stripPunctuation(doc['title'])
        data=stripped_desc_list+stripped_title_list
        if word in data:
            df+=1
    return df
    
def findTermFrequency(key,doc):
    """
    Calculate Term Frequencies(#times a word appears in a particular document) for all the words
    and documents
    """
    if type(doc) is list:
        return doc.count(key)
    stripped_title_list=stripPunctuation(doc['title'])
    stripped_desc_list=stripPunctuation(doc['description'])
    data=stripped_title_list+stripped_desc_list
    return data.count(key)

def calculateDocVector(doc,ordered_dict,all_docs):
    """
    Calculate all Document vectors using all words present in the Ordered Dictionary
    """
    doc_vec=[]
    for key,val in ordered_dict.iteritems():
        doc_freq=findDocFrequency(key,all_docs)
        term_freq=findTermFrequency(key,doc)
        if doc_freq:
            wt=term_freq*(math.log10(float(len(all_docs))/float(doc_freq)))
        else:
            wt=0
        doc_vec.append(wt)
    return doc_vec
    
def calculateQueryVector(queryList,ordered_dict,all_docs):
    """
    Calculate the Query vectors using all words present in the Ordered Dictionary
    """
    query_vec=[]    
    for key,val in ordered_dict.iteritems():
        doc_freq=findDocFrequency(key,all_docs)
        term_freq=findTermFrequency(key,queryList)
        if doc_freq:
            wt=term_freq*(math.log10(float(len(all_docs))/float(doc_freq)))
        else:
            wt=0
        query_vec.append(wt)
    return query_vec

def getUserFeedback(results,dvec):
    """
    Get Relevance Feedback from the user and create separate collections for relevant
    and non-relevant documents
    """
    rel_docs={}
    rel_docs['docs']=[]
    rel_docs['dvec']=[]
    nrel_docs={}
    nrel_docs['docs']=[]
    nrel_docs['dvec']=[]
    nrelevant=0
    print ('Please provide feedback about relevance of the documents\n')
    for doc in results:
        print '-------------------------------------------'
        print 'DOC RANK : ', doc['rank']
        print 'DOC URL : ', doc['url'].encode('ascii','ignore')
        print 'DOC TITLE : ', doc['title'].encode('ascii','ignore')
        print 'DOC DESCRIPTION : ', doc['description'].encode('ascii','ignore')
        print '-------------------------------------------'
        feedback=raw_input('Please Enter [1] : Relevant  [2] : Non Relevant ')
        print 
        if(int(feedback)==1):
            rel_docs['docs'].append(doc)
            rel_docs['dvec'].append(dvec[doc['url']])
            nrelevant+=1
        else:
            nrel_docs['docs'].append(doc)
            nrel_docs['dvec'].append(np.array(dvec[doc['url']])) 
    return (rel_docs,nrel_docs,float(nrelevant)/float(len(results)))
    
def roccioAlgorithm(q,r,nr):
    """
    Implementing ROCCIO Algorithm for query vector reweighting
    """
    alpha=1
    beta=0.70
    gamma=0.15
    q=np.array(q)
    new_query_vec=(alpha*q)+((beta/len(r['dvec']))*np.sum(r['dvec'],axis=0))-((gamma/len(nr['dvec']))*np.sum(nr['dvec'],axis=0))
    return list(new_query_vec)
    
def selectNewWords(new_list,query_words,tagged_words):
    """
    Select new words to append to the initial query
    """
    ratio_range=(1,1.5)
    new_nounwords=[]
    for i,val in enumerate(tagged_words):
        tagged_words[i][0].encode('utf-8')
        
    noun_list=[v[0].encode('utf-8') for i, v in enumerate(tagged_words) if (re.search(r"^NN", v[1]) or re.search(r"^JJ", v[1]))]
    for each in new_list:
        if each[0] not in query_words and each[0] in noun_list:
            new_nounwords.append(each)
            
    new_selected_words=[]
    new_selected_words.append(new_nounwords[0][0])
    if(ratio_range[0]<=(float(new_nounwords[0][1])/float(new_nounwords[1][1]))<ratio_range[1] and new_nounwords[1][1] != new_nounwords[2][1]):
        new_selected_words.append(new_nounwords[1][0])
    return new_selected_words

    
def wordTaggerFunction(docs):
    """
    This function uses the NLTK tagger to tokenize the description of documents and 
    create a list of tagged words.
    """
    tagged_list=[]
    for each in docs:
        text=nltk.word_tokenize(each['description'])
        tagged_list.append(nltk.pos_tag(text))
    return sum(tagged_list, [])
    
def main():
    print
    query=raw_input('Enter the Query : ')
    precisionValue=raw_input('Enter the desired Precision@10 value : ') #Precision value to be achieved eventually
    run_count=0
    word_dictionary={} # Collection of all words extracted from the BING Search results
    achieved_precision=0
    while(achieved_precision < precisionValue): #Run until the desired precision is achieved      
        query=urllib.quote(query)
        queryList=query.split('%20')
        bingUrl = 'https://api.datamarket.azure.com/Bing/Search/Web?Query=%27userquery%27&$top=10&$format=json'    
        accountKey = "JbyKIOD9ljIg7tO7i8C4PzOBmKzTuUOcjCIA9R53A8k" #BING account key
        bingUrl=bingUrl.replace('userquery',query) 
        content=bing_api(bingUrl,accountKey)
        queryResults=json.loads(content) #Parse the json documents containing query results
        
        print 'FEEDBACK SUMMARY'
        print 'Query : ',query
        print 'BingUrl : ',bingUrl.encode('ascii','ignore')
        print 'Number of Results : ',len(queryResults['d']['results'])     
        if len(queryResults['d']['results']) < 10: # Terminating the program if search results are not sufficient  
            print ('Not Enough Query search results..Terminating the program')
            break
            
        qResultDict={}
        qResultDict['data']=[]
        for (rank,each) in enumerate(queryResults['d']['results']):
            tempDict={}
            tempDict['rank']=rank+1
            tempDict['title']=each['Title'].encode('ascii','ignore').lower()
            tempDict['url']=each['DisplayUrl'].encode('ascii','ignore')
            tempDict['description']=each['Description'].encode('ascii','ignore').lower()
            qResultDict['data'].append(tempDict)
        word_dictionary=updateDictionary(word_dictionary,qResultDict['data']) #Contains the dictionary of all words along with 
                                                                              # their counts 
        
        #Create a list of all document vectors
        doc_vectors={} 
        for doc in qResultDict['data']:
            doc_vectors[doc['url']]=calculateDocVector(doc,word_dictionary,qResultDict['data'])
            
        #creating query vector
        query_vector=calculateQueryVector(queryList,word_dictionary,qResultDict['data'])
        
        #Calculating the relevant , non relevant documents based on user feedback and calculating
        #the precision achieved by the algorithm based on the same
        (rel_docs,nrel_docs,achieved_precision)=getUserFeedback(qResultDict['data'],doc_vectors)
        print "Precision Achieved based on user feedback : ",achieved_precision    
        
        # Terminate the algorithm when the desired precision is achieved 
        if(float(achieved_precision) >= float(precisionValue)):
            print ('Required precision value achieved..Algorithm terminates')
            break
      
        # Terminate the algorithm if no relevant documents are present in the search results
        if(len(rel_docs['docs'])==0 and run_count==0):
            print ('No relevant results in the initial search query..Quitting')
            break
        
    
        run_count+=1 #Contains the number of runs required by the algorithm to achieve the desired precision value  
        
        #Calculating the new query vector using ROCCIOs Algorithm and selecting all positive entries present in it.
        new_query_vec=roccioAlgorithm(query_vector,rel_docs,nrel_docs)
        new_query_vec=list(np.array(new_query_vec).clip(min=0))
        
        i=0
        all_new_words=[] #Contains a list of all possible words that can be appended to the Query 
        for key,val in word_dictionary.iteritems():
            if(new_query_vec[i] > 0):
                all_new_words.append((key,new_query_vec[i]))
            i+=1 
            
        #Sorting the possible words in decreasing order of their weights calculated by ROCCIOs algorithm 
        all_new_words.sort(key=lambda tup:tup[1],reverse=True)
    
        #Contains the WORD-TAG values 
        tagged_words=wordTaggerFunction(qResultDict['data'])
        
        #Get the two new words to be appended to the former user query
        two_new_words=selectNewWords(all_new_words,queryList,tagged_words)
        two_new_words=" ".join(two_new_words)
        
        #Modify the initial user query by appending the newly selected words.
        query=urllib.unquote(query)+" "+two_new_words
        print "Modified Query : ",query
        print
    #Print the number of iterations required to achieve the desired precision value 
    print 'RUN COUNT : ',run_count
    
if __name__ == '__main__':
    main()