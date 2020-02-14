#Track word frequency
from collections import Counter
#Numpy is used for arrays for calculating edit distance
import numpy as np
#Native Python package
import random
#A third party substitution list to expand contractions eg I'm, imma -> I am
import contractions
#Identifies the likely language of a text string with a confidence score
import langid
from langid.langid import LanguageIdentifier, model
#Used for edit distance calculations
import nltk
#Regex is faster and more full featured than the native Python regex
import regex
#Filehandling
import pickle
#Sortedcontainers provides faster searching and indexing than python lists
from sortedcontainers import SortedList

#################
### FUNCTIONS ###
#################

#strip username references of tweets (eg @USER12345678)
def remove_user_ids(tweets):
    rule = regex.compile(r"@USER_\w{8}")
    tweets = '\n'.join(tweets)
    tweets = rule.sub(r" ",tweets)
    return tweets.splitlines()


#detect language of the tweets, exclude other items that are probably in another language
def remove_foreign_tweets(tweets):
    english = []
    foreign = []
    classifier = LanguageIdentifier.from_modelstring(model, norm_probs = True) #setup classifier

    for tweet in tweets:
        tweet_class = classifier.classify(tweet)
        if tweet_class[0] != 'en' and tweet_class[1] > 0.99: #Other languages need higher confidence to avoid coincidental overfitting
            foreign.append(tweet)
        elif tweet_class[0] == 'es' and tweet_class[1] > 0.95: #Spanish tweets are common in US due to Hispanic influence so their confidence is set to 0.95
            foreign.append(tweet)
        else:
            english.append(tweet)
    #UNIT TEST
    #pwrite('nonenglish', nonenglish)
    #pwrite('english', english)
    return english

#Replace repeated character sequences of length 3 or greater with length 1.
def remove_repeated_characters(tweets):
    tweets = '\n'.join(tweets)

    rule = regex.compile(r"\b\w*(\w)\1{2,}\w*\b")
    tweets = rule.sub(r" ", tweets)

    rule = regex.compile(r"\b\w*(\w\w)\1{2,}\w*\b")
    tweets = rule.sub(r" ", tweets)

    rule = regex.compile(r"\b\w*(\w\w\w)\1{2,}\w*\b")
    tweets = rule.sub(r" ", tweets)

    return tweets.splitlines()

def fix_contractions(tweets):
    tweets = '\n'.join(tweets)
    tweets = contractions.fix(tweets)
    return tweets.splitlines()

#A word at the start of a sentence or line is capitalised but it may not be a
#proper noun. This function flags such words by appending YYY (this is a bit of
#a performance workaround because of regex's inability to replace a word with
#all upper case efficiently)
def flag_ambiguous_capitalisation(tweets):
    tweets = '\n'.join(tweets)
    rule = regex.compile(r"([\n\.\!\?])\s*(\w+)")
    tweets = rule.sub(r" \2YYY ", tweets) #YYY is a placeholder to identify ambiguous words

    return tweets.splitlines()

def remove_hashtags(tweets):
    tweets = '\n'.join(tweets)
    rule = regex.compile(r"#\S*")
    tweets = rule.sub(' ', tweets)

    return tweets.splitlines()

def remove_symbols_numbers(tweets):
    tweets = '\n'.join(tweets)
    rule = regex.compile(r"[A-Za-z]+[^A-Za-z\s?!,.-:;()\'\"/]+[A-Za-z]+")
    tweets = rule.sub(' ', tweets)

    rule = regex.compile(r"[^A-Za-z\s]")
    tweets = rule.sub(' ', tweets)

    return tweets.splitlines()

# Words are capitalised to indicate one of three possible categories
# 1) common noun (lowercase) 2) proper noun (Uppercase) 3) Unsure (ALL CAPS)
def capitalise_by_flags(words):
    for i, v in enumerate(words):
        if v[-3:] == 'YYY':
            words[i] = v[0:-3].upper()
        elif v.isupper() == False and v.islower() == False:
            if v[1:].islower() == False:
                words[i] = v.upper()
    return words

#Very short words will have far too many possible sources and cannot be
#Realistically handled by this algorithm
def remove_short_words(count_candidates, min_candidate_length = 4):
    if min_candidate_length < 2:
        return count_candidates
    for v in list(count_candidates.keys()):
        if len(v) < min_candidate_length:
            count_candidates.pop(v)
    return count_candidates

#Words that are only used very infrequently on Twitter are probably typos or
#not in common use.
def remove_infrequent_words(count_candidates, min_cand_occurrences = 0):
    if min_cand_occurrences == 0:
        return count_candidates
    for v in list(count_candidates.keys()):
        if count_candidates[v] < min_cand_occurrences:
            count_candidates.pop(v)
    return count_candidates

#The dictionary is full of obscure words. Obscure words tend not to be used as
#blend source words. Removing these obscure words improves matching.
def remove_infrequent_dict_words(count_dictionary, min_dict_occurrences = 0):
    if min_dict_occurrences == -1:
        return count_dictionary
    for v in list(count_dictionary.keys()):
        if count_dictionary[v] < min_dict_occurrences:
           count_dictionary.pop(v)
    return count_dictionary

#As proper nouns are not in the dictionary, including proper nouns in consideration
#results in too many false positives. Removing proper nouns improves matching
def remove_likely_proper_nouns(count_candidates, count_candidates_case, proper_noun_cutoff = 0.8):
    table_candidates_case = list(count_candidates.keys())

    for i, v in enumerate(table_candidates_case):
        table_candidates_case[i] = [v,0,0,0]
        table_candidates_case[i][1] = count_candidates_case.get(v,0)
        table_candidates_case[i][2] = count_candidates_case.get(v[0].upper() + v[1:], 0)
        table_candidates_case[i][3] = count_candidates_case.get(v.upper(),0)

    for i,v in enumerate(table_candidates_case):
        if v[1] == 0 and v[2] > 0:
            count_candidates.pop(v[0])
        elif v[1] != 0 and v[2] != 0:
            if (v[2]/(v[1]+v[2])) > proper_noun_cutoff and (v[1]+v[2]+v[3])>=10:
                count_candidates.pop(v[0])

    return count_candidates

#Words that are a combination of two words eg racecar are not blend words, but
#are often not in the dictionary. Removing these improves matching. This is a
#performance intensive function
def remove_likely_compound_words(count_candidates, count_dictionary):
    sorted_candidates = SortedList(count_candidates.keys())
    sorted_dictionary = SortedList(count_dictionary.keys())
    compound_candidates = []

    for i,v in enumerate(sorted_candidates):
        if i % 100 == 0:
            print(i)
        if len(v) >= 6:
            #by only searching through a slice a sorted list for matching words
            #performance is greatly improved
            slice_start = v[0:3]
            slice_end = v[0:3] + 'zzz'
            for w1 in sorted_dictionary.irange(slice_start,slice_end):
                if (len(w1) >= 3 and len(w1) < (len(v) - 3)):
                    w2 = v[len(w1):]
                    if w2 in sorted_dictionary:
                        count_candidates.pop(v)
                        compound_candidates.append(v)
                        break
        #print(compound_candidates)
    return count_candidates

#An infrequent word that is very similar to a frequent word is categorised
#as a likely typo.
def remove_likely_typos(count_candidates, count_dictionary):
    matchlist = []
    for v in list(count_candidates.keys()):
        if len(v) <= 4:
            distance = 0
        elif len(v) <= 12:
            distance = 1
        else:
            distance = 2
        if distance == 0:
            continue
        for j in count_dictionary:
            if ((abs(len(j) - len(v))) <= distance
            and count_dictionary[j] >= (10*count_candidates[v])
            and (nltk.edit_distance(j, v) <= distance)):
                count_candidates.pop(v)
                matchlist.append([v,j])
                #print(matchlist[-1])
                break
    return count_candidates

#All possible blend words are compared to possible source word combiantions.
#These are stored in an array of up to 25 source words. A modified local edit
#distance algorithm is used to value words that have a lot of overlap.
def find_likely_source_words(count_candidates, count_dictionary):

    candidates_w_sources = dict()
    for v in count_candidates.keys():
        candidates_w_sources[v] = [SortedList(),SortedList()]

    sorted_dictionary = SortedList()
    rev_sorted_dictionary = SortedList()
    for v in count_dictionary.keys():
        sorted_dictionary.add(v)
        rev_sorted_dictionary.add(v[::-1])

    for v in candidates_w_sources:
        counter = 0
        len_v = len(v)
        prefix = v[0:2]
        search_slice = sorted_dictionary.irange(prefix,str(prefix+'zzz'))
        for source in search_slice:
            score = local_distance(source, v) / len_v
            if counter == 25:
                if score > candidates_w_sources[v][0][0][0]:
                    candidates_w_sources[v][0].pop(0)
                    candidates_w_sources[v][0].add([score,source])
            else:
                candidates_w_sources[v][0].add([score,source])
                counter += 1

        counter = 0
        len_v = len(v)
        suffix = v[-1:-3:-1]
        vback = v[::-1]
        search_slice = rev_sorted_dictionary.irange(suffix,str(suffix+'zzz'))
        for source in search_slice:
            score = local_distance(source,vback) / len_v
            if counter == 25:
                if score > candidates_w_sources[v][1][0][0]:
                    candidates_w_sources[v][1].pop(0)
                    candidates_w_sources[v][1].add([score,source[::-1]])
            else:
                candidates_w_sources[v][1].add([score,source[::-1]])
                counter += 1

    for v in list(candidates_w_sources.keys()):
        if (len(candidates_w_sources[v][0]) == 0
        or len(candidates_w_sources[v][1]) == 0
        or candidates_w_sources[v][0][-1][0] == 1
        or candidates_w_sources[v][1][-1][0] == 1):
            candidates_w_sources.pop(v)

    return candidates_w_sources

#Reciprocal Mean Rank is used to assess how close the algorithms guess for
#source words was to the true source word.
def rank_performance_against_true_answers(candidates_w_sources, true_answers):
    scored_results = dict()

    for v in true_answers:
        if v[0] in candidates_w_sources:
                answer = v[0]

                start_rank = 0
                start_word = v[1]

                for n, m in enumerate(candidates_w_sources[v[0]][0]):
                    if start_word == m[1]:
                        start_rank = 1/(len(candidates_w_sources[v[0]][0]) - n)
                        break

                end_rank = 0
                end_word = v[2]

                for n, m in enumerate(candidates_w_sources[v[0]][1]):
                    if end_word == m[1]:
                        end_rank = 1/(len(candidates_w_sources[v[0]][1]) - n)
                        break

                scored_results[v[0]] = (start_rank + end_rank) / 2

    return scored_results



######################
### SUPPORTING CODE ##
######################

### UNIT TEST CODE ###
#This can compare the lists at any two points in the candidate selection process
#to monitor removal of items and performance.
def unit_test(before, after, sample_size = 10, true_items = None):
    if isinstance(before, dict):
        before = before.keys()
    if isinstance(after, dict):
        after = after.keys()

    total = before.length()
    total_after = after.length()
    print("Original Size: " + total )
    print("Items Cut: " + total_after )

    if true_source_words is not None:

        positives = list(set(before) & set(true_items))
        true_positives = list(set(after) & set(true_items))
        false_positives = list(set(after) - set(true_items))

        print("Positive items cut: " + (positives.length() - true_positives.length()))
        print("True positive rate: " + (true_positives.length() / positives.length()))

        negatives = list(set(before) - set(true_items)).length()
        true_negatives = \
            list(
                set(
                    list(set(before) - set(after))
                    - set(true_items)
                )
            )
        false_negatives = \
            list(
                set(
                    list(set(before) - set(after))
                    & set(true_items)
                )
            )
        print("Negative items cut: " + (true_negatives.length()))
        print("True negative rate: " + (true_negatives.length() / negatives.length()))

        print("Sample true positives: " + random.sample(true_positives, sample_size))
        print("Sample false positives: " + random.sample(false_positives, sample_size))
        print("Sample true negatives: " + random.sample(true_negatives, sample_size))
        print("Sample false negatives: " + random.sample(false_negatives, sample_size))

    else:
        print("Sample kept: " + random.sample(dict_after, sample_size))
        print("Sample cut: " + random.sample(list(set(dict_before) * set(dict_after)), sample_size))

    return

### EDIT DISTANCE CALCULATIONS ####
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

### FILE READING FUNCTIONS ###
def read(filename,code='utf8'):
    #uses native python methds to read a file as string
    f = open(filename,'r',encoding=code)
    data = f.read()
    f.close
    return data
def pread(filename):
    #Uses the pickle module to read from a pickle file as python objects
    f = open(filename,'rb')
    data = pickle.load(f)
    f.close()
    return data
def pwrite(filename, data):
    #Uses the pickle module to write to a pickle file as python objects
    f = open(filename,'wb')
    pickle.dump(data,f)
    f.close()
    return

#################
### MAIN CODE ###
#################

#Read a plaintext list of tweets, each seperated by 1 line (no metadata)
tweets = read('.\\data\\tweets.txt').splitlines()

#Read a plaintext list of known words, each seperated by 1 line.
#These will be used as potential source words for unknown blends
dictionary = read('.\\data\\dictionary.txt').splitlines()

#A list of true blends (not in dictionary) can be provided for unit testing
#And algorithm performance assessment. Use a plaintext file in the following
#Format, with each new entry on a new line
#BLEND_WORD START_SOURCE_WORD END_SOURCE_WORD
#eg "belieber believe Bieber"
true_answers = read('.\\data\\optional_answers.txt').splitlines()
for i,v in enumerate(true_answers):
    true_answers[i] = v.split()
print("remove_user_ids")
tweets = remove_user_ids(tweets)
print("remove_foreign_tweets")
tweets = remove_foreign_tweets(tweets)
print("remove_repeated_characters")
tweets = remove_repeated_characters(tweets)
print("fix_contractions")
tweets = fix_contractions(tweets)
print("flag_ambiguous_capitalisation")
tweets = flag_ambiguous_capitalisation(tweets)
print("remove_hashtags")
tweets = remove_hashtags(tweets)
print("remove_symbols_numbers")
tweets = remove_symbols_numbers(tweets)
words = (' '.join(tweets)).split()
print("capitalise_by_flags")
words = capitalise_by_flags(words)

#data formatting
print("#format data for next functions")
count_candidates = Counter(v.lower() for v in words)
count_candidates_case = Counter(words)
count_dictionary = dict()
for v in dictionary:
    count_dictionary[v] = count_candidates.get(v,0)
	
print("remove_short_words")
count_candidates = remove_short_words(count_candidates)
print("remove_infrequent_words")
count_candidates = remove_infrequent_words(count_candidates)
print("remove_infrequent_dict_words")
count_dictionary = remove_infrequent_dict_words(count_dictionary)
print("remove_likely_proper_nouns")
count_candidates = remove_likely_proper_nouns(count_candidates,count_candidates_case)
print("remove_likely_compound_words")
count_candidates = remove_likely_compound_words(count_candidates, count_dictionary)
print("remove_likely_typos")
count_candidates = remove_likely_typos(count_candidates, count_dictionary)
print("find_likely_source_words")
candidates_w_sources = find_likely_source_words(count_candidates, count_dictionary)
pwrite('results\\candidates_w_sources.txt',list(candidates_w_sources.items()))

if true_answers is not None:
    print("rank_performance_against_true_answers")
    scored_results = rank_performance_against_true_answers(candidates_w_sources, true_answers)
    pwrite('results\\rank_performance_against_true_answers.txt', scored_results)
