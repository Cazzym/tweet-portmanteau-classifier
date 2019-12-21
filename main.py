import nltk #NLTK is used for edit distance calculations
import contractions #Contractions is used to expand contractions eg I'm, imma -> I am
import regex #regex is faster and more full featured than the native Python re
import numpy as np #numpy is used for arrays for calculating edit distance
from sortedcontainers import SortedList #sortedcontainers provides faster searching and indexing than python lists
import pickle #pickle allows simple file handling data to be passed between functions, results reviewed and debug
import random
import langid
from langid.langid import LanguageIdentifier, model

def read(filename,code='utf8'):
    #uses native python methds to read a file as string
    f = open(filename + '.txt','r',encoding=code)
    data = f.read()
    f.close
    return data
def pread(filename):
    #Uses the pickle module to read from a pickle file as python objects
    f = open(filename + '.txt','rb')
    data = pickle.load(f)
    f.close()
    return data
def pwrite(filename, data):
    #Uses the pickle module to write to a pickle file as python objects
    f = open(filename + '.txt','wb')
    pickle.dump(data,f)
    f.close()
    return

def performance_reporting(candidate_list,dict_list): #This function reports remaining candidates and source words to assess screening performance

    candidates=[]
    for i,v in enumerate(candidate_list):
        candidates.append(v[0])
    sorteddict = SortedList(dict_list)
    blends = [line.rstrip('\n') for line in open('blendscorrected.txt')]
    blendsonly = []
    for i, v in enumerate(blends):
        blends[i] = v.split()
        blendsonly.append(blends[i][0])
    missingblends = []
    for v in blendsonly:
        if v not in candidates:
            missingblends.append(v)

    missingsources = []
    for i, v in enumerate(blends):
        if sorteddict.__contains__(v[1]) == False:
            missingsources.append(v[1])
        if sorteddict.__contains__(v[2]) == False:
            missingsources.append(v[2])

    print("Candidate list:")
    print(len(candidates))
    print("Reduction:")
    print(1-(len(candidates)/17538))
    print("Dictionary list:")
    print(len(sorteddict))
    print("Reduction:")
    print(1-(len(sorteddict)/370059))

#    print("Missing Sources:")
#    for v in missingsources:
#        print(v)

#    print("Missing Blends:")
#    for v in missingblends:
#        print(v)

    print("Missing blends: " + str(len(missingblends)))
    print(len(missingblends)/len(blends))
    print("Missing sources: " + str(len(missingsources)))
    print(len(missingsources)/(2*len(blends)))
    return

def local_distance(i, j):
    li = len(i)+1
    lj = len(j)+1
    A = np.zeros([li,lj])
    for h in range(1, li):
        for k in range(1,lj):
            A[h][0] = -h
            A[0][k] = -k
    for h in range(1, li):
        for k in range (1, lj):
            A[h,k] = max(
                (A[h-1,k-1]+(1 if i[h-1]==j[k-1] else -1)),
                (A[h][k-1] - 1),
                (A[h-1][k] - 1)
            )
    return A.max()

def lang_detection(): #detect language of the tweets, exclude like other items
    tweets = read('tweetsansi','ansi') #read tweet file (changed to ansi to allow readability)
    tweets = tweets.splitlines() #convert to list
    langid = LanguageIdentifier.from_modelstring(model, norm_probs=True) #setup classifier
    english=[]
    nonenglish=[]
    pattern = regex.compile(r"@USER_\w{8}") #strip usernames before classifiying

    print('Classifying languages...')
    for i,v in enumerate(tweets):
        v = pattern.sub(r" ",v)
        lang = langid.classify(v)
        if lang[0] == 'es' and lang[1] > 0.95: #Spanish tweets are common so their confidence is set to 0.95
            nonenglish.append(tweets[i])
        elif lang[0] != 'en' and lang[1] >0.99: #Other languages need higher confidence to avoid coincidental overfitting
            nonenglish.append(tweets[i])
        else:
            english.append(tweets[i])
    pwrite('nonenglish', nonenglish)
    pwrite('english', english)
    return

def preprocessing_basic(): #This function does a variety of preprocessing
    tweets = pread('english')
    tweets = "\n".join(tweets) #convert to list
    dict = read('dict')
    dict = SortedList(dict.split())
    pwrite('dictpython',dict)

    print("Removing repeated characters...") #Replace repeated character sequences of length 3 or greater with length 1.
    pattern = regex.compile(r"\b\w*(\w)\1{2,}\w*\b")
    tweets = pattern.sub(r" ", tweets)
    pattern = regex.compile(r"\b\w*(\w\w)\1{2,}\w*\b")
    tweets = pattern.sub(r" ", tweets)
    pattern = regex.compile(r"\b\w*(\w\w\w)\1{2,}\w*\b")
    tweets = pattern.sub(r" ", tweets)

    print("Expanding contractions...") #uses the contractions package to expand I'm -> I am etc.
    tweets = contractions.fix(tweets)

    print("Classifying ambigious capitals...") #A word at the start of a sentence or line is capitalised but it may not be a proper noun. This function flags such words by appending YYY (this is a bit of a workaround because of regex's inability to replace a word with all upper case efficiently)
    pattern = regex.compile(r"([\n\.\!\?]|@USER_\w{8})\s*(\w+)")
    tweets = pattern.sub(r" \2YYY ", tweets) #YYY is a placeholder to identify ambiguous words

    #originally this code was used to make words all uppercase but it was incredibly slow (1 hour)
    #stlst = pattern.findall(tweets)
    #for i in stlst:
    #    j = regex.escape(i)
    #    h = j.upper()
    #    tweets = regex.sub(j,h,tweets)

    print("Removing hashtags...")
    pattern = regex.compile(r"#\S*")
    tweets = pattern.sub(' ', tweets)

    print("Removing words with symbols or numbers...")
    pattern = regex.compile(r"[A-Za-z]+[^A-Za-z\s?!,.-:;()\'\"/]+[A-Za-z]+")
    tweets = pattern.sub(' ', tweets)

    print("Removing symbols...")
    pattern = regex.compile(r"[^A-Za-z\s]")
    tweets = pattern.sub(' ', tweets)

    print("Standardising white...")
    tweets = ' '.join(tweets.split())

    print("Standardising capitalisation...") #lowercase stay lowercase, first letter capitalised stay first letter capitalised, ambigious changed to all caps
    tweets = tweets.split()
    for i, v in enumerate(tweets):
        if v[-3:] == 'YYY':
            tweets[i] = v[0:-3].upper()
        elif v.isupper() == False and v.islower() == False:
            if v[1:].islower() == False:
                tweets[i] = v.upper()
    tweets = sorted(tweets, key=lambda s: s.lower()) #Case insensitive sort

    tweetsfull = [] #Tweetsfull list is an untouched, lower case list used to tally words on dictionary
    for v in tweets:
        tweetsfull.append(v.lower())
    tweetsfull = sorted(tweetsfull)
    sortedtweetsfull=SortedList(tweetsfull)
    dict=list(dict)
    sorteddict=SortedList(dict)
    word = ''
    print("Counting occurrences...")
    for i, v in enumerate(dict):
        dict[i] = [v,0]
    for i, v in enumerate(tweetsfull): #Count the number of times each word occurs, and add it to the dictionary list
        if word != v:
            word = v
            if sorteddict.__contains__(v):
                j = sortedtweetsfull.count(v)
                k = sorteddict.index(v)
                dict[k][1] += j

    print('Removing dictionary words from candidates...')
    removelist = []
    for i,v in enumerate(tweets):
        if sorteddict.__contains__(v.lower()):
            removelist.append(i)
    for v in sorted(removelist,reverse=True):
        tweets.pop(v)

    pwrite('dictcount1',dict)
    pwrite('preprocessing_basic',tweets)
    pwrite('preprocessing_basic_full',tweetsfull)

    return

def preprocessing_aggressive(min_candidate_length = 4,min_dict_occurrences=1, min_cand_occurrences=0,proper_noun_cutoff=0.8):
    tweets = pread('preprocessing_basic')
    tweetsfull = pread('preprocessing_basic_full')
    dict = pread('dictcount1')

    print("Removing short words...") #cut words of 1-3 letters
    removelist = []
    for i, v in enumerate(tweets):
        if len(v) < min_candidate_length:
            removelist.append(i)
    for v in sorted(removelist,reverse=True):
        tweets.pop(v)

    word = ''
    sortedtweetsfull = SortedList(tweetsfull) #The sorted containers object is faster to search than the standard python list
    sorteddict = SortedList(dict) #The sorted containers object is faster to search than the standard python list

    print("Removing infrequent occurrences from dictionary...")
    removelist = []
    for i,v in enumerate(dict):
        if v[1] < min_dict_occurrences:
           removelist.append(i)
    for v in sorted(removelist,reverse=True):
        dict.pop(v)
    sorteddict.clear()
    for v in dict:
        sorteddict.add(v[0])

    tweetscount = []
    word = ''
    low = 0
    upp = 0
    ambig = 0

    print("Counting cases and removing infrequent candidates...")
    candidates = []

    for i,v in enumerate(tweets):
        if word != v.lower():
            if (low+upp+ambig) >= min_cand_occurrences:
                candidates.append([word,low,upp,ambig])
            word = v.lower()
            low = 0
            upp = 0
            ambig = 0
        if v.islower():
            low += 1
        elif v.isupper():
            ambig += 1
        else:
            upp += 1
    print('Removing proper nouns from candidates...')
    removelist = []
    for i,v in enumerate(candidates):
        if v[1] == 0 and v[2] > 0:
            removelist.append(i)
        elif v[1] != 0 and v[2] != 0:
            if (v[2]/(v[1]+v[2])) > proper_noun_cutoff and (v[1]+v[2]+v[3])>=10:
                removelist.append(i)
    for v in sorted(removelist,reverse=True):
        candidates.pop(v)

    exitloop = False
    candidatesonly = []
    removelist = []
    compoundlist = []
    print('Removing compound words...')
    for i,v in enumerate(candidates):
        candidatesonly.append(v[0])
    for i,v in enumerate(candidatesonly):
        if len(v) >= 6:
            slice = v[0:3]
            sr = sorteddict.irange(slice,str(slice+'zzz'))
            for h in sr:
                if exitloop == True:
                    exitloop = False
                    break
                if len(h) >= 3:
                    for l,m in enumerate(sorteddict):
                        if len(m) >=3:
                            if (h + m) == v:
                                removelist.append(i)
                                compoundlist.append(v)
                                exitloop = True
                                break
    for v in sorted(removelist,reverse=True):
        candidatesonly.pop(v)
    removelist = []
    for i,v in enumerate(candidates):
        if v[0] not in candidatesonly:
            removelist.append(i)
    for v in sorted(removelist,reverse=True):
        candidates.pop(v)

#    pwrite('readycandidatesonly',candidatesonly)

    pwrite('readycandidates',candidates)
    pwrite('dictcount',dict)
    pwrite('dictsorted',sorteddict)

    return

def scoring_algorithm():
    candidatesonly = pread('readycandidatesonly')
    candidates = pread('readycandidates')
    dict = pread('dictcount')
    sorteddict = pread('dictsorted')
    candidates = sorted(candidates)

    removelist = []
    matchlist = []
#This typo removal formula was too aggressive, and thus switched off.
#    for i, v in enumerate(candidates):
#        if len(v[0]) <= 4:
#            distance = 0
#        elif len(v[0]) <= 12:
#            distance = 1
#        else:
#            distance = 2
#        for h, j in enumerate(dict):
#            if (abs(len(j[0]) - len(v[0])) <= distance and j[1] >= (10*v[1]) and (nltk.edit_distance(j[0], v[0])) <= distance):
#                removelist.append(i)
#                matchlist.append([v[0],j[0]])
#                print(matchlist[-1])
#                break
#    for v in sorted(removelist,reverse=True):
#        candidates.pop(v)


    removelist =[]
    for i,v in enumerate(candidates):
        candidates[i] = [v[0],(v[1]+v[2]+v[3])]
    for i,v in enumerate(candidates):
        counter = 0
        lv = len(v[0])
        candidates[i].append(SortedList())
        slice = v[0][0:2]
        sr = sorteddict.irange(slice,str(slice+'zzz'))
        for h in sr:
            score = local_distance(h,v[0])/lv
            if counter == 25:
                if score > candidates[i][2][0][0]:
                    candidates[i][2].pop(0)
                    candidates[i][2].add([score,h])
            else:
                candidates[i][2].add([score,h])
                counter += 1
    revsorteddict = SortedList()
    for v in sorteddict:
        revsorteddict.add(v[::-1])
    for i,v in enumerate(candidates):
        counter = 0
        lv = len(v[0])
        candidates[i].append(SortedList())
        slice = v[0][-1:-3:-1]
        vback = v[0][::-1]
        sr = revsorteddict.irange(slice,str(slice+'zzz'))
        for h in sr:
            score = local_distance(h,vback)/lv
            if counter == 25:
                if score > candidates[i][3][0][0]:
                    candidates[i][3].pop(0)
                    candidates[i][3].add([score,h])
            else:
                candidates[i][3].add([score,h])
                counter += 1
        print(candidates[i])
    print("Removing items")
    for i,v in enumerate(candidates):
        if ((v[2][-1][0] == 1) if len(v[2])>0 else (1 == 0)) or ((v[3][-1][0] == 1 ) if len(v[3])>0 else (1 == 0)):
            removelist.append(i)
    for v in sorted(removelist,reverse=True):
        print (candidates[v][0]+((str(candidates[v][2][-1])) if len(candidates[v][2])>0  else 'NA')+(str(candidates[v][3][-1]) if len(candidates[v][3])>0  else 'NA'))
        candidates.pop(v)

    pwrite('scoredcandidates',candidates)

    return

def mean_recip_rank():
    candidates = pread('scoredcandidates')
    dict = pread('dictcount')
    sorteddict = pread('dictsorted')
    blends = [line.rstrip('\n') for line in open('blendscorrected.txt')]
    foundblends = []
    inversemeanranks = []

    for i,v in enumerate(blends):
        blends[i] = v.split()
    for i,v in enumerate(blends):
        found = False
        for h,j in enumerate(candidates):
            if v[0] == j[0]:
                found = True
                break
        if found == True:
            foundblends.append(v[0])
            rank1 = 0
            rank2 = 0
            for n,m in enumerate(candidates[h][2]):
                if v[1] == m[1]:
                    rank1 = 1/(len(candidates[h][2]) - n)
                    break
            for o,p in enumerate(candidates[h][3]):
                if v[1] == p[1]:
                    rank2 = 1/(len(candidates[h][3]) - o)
                    break
            rank = (rank1 + rank2)/2
            inversemeanranks.append([rank,v[0]])
    print(inversemeanranks)
    print(len(inversemeanranks))
    print(len(candidates))

    return


lang_detection()
preprocessing_basic()
preprocessing_aggressive()
scoring_algorithm()
mean_recip_rank()
