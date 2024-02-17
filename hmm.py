import pandas as pd
import numpy as np
import re

# Question 2a
def q2a():
    output_file_name ='output_probs.txt'
    train_file = open('twitter_train.txt', 'r', encoding='utf8')
    tag_file = open('twitter_tags.txt', 'r', encoding='utf8')
    output_file = open(output_file_name, 'w', encoding='utf8')

    train_lines = train_file.readlines()
    tag_lines = tag_file.readlines()

    num_of_tags = dict() # {"@": 10}
    num_of_tags_and_tokens  = dict()

    wordList = set()

    output_file.write("Token Tag Prob \n")

    for line in tag_lines:
        tag = line.split()[0]
        num_of_tags[tag] = 0
        num_of_tags_and_tokens[tag] = {}

    for line in train_lines:
        token_tag_pair = line.split()
        if len(token_tag_pair) == 0:
            continue
        token = token_tag_pair[0].upper()
        if token[0:6] == "@USER_":
            token = "@USER_"
        tag = token_tag_pair[1]

        num_of_tags[tag] += 1

        if token not in num_of_tags_and_tokens[tag]:
            num_of_tags_and_tokens[tag][token] = 1
        else:
            num_of_tags_and_tokens[tag][token] += 1

        wordList.add(token)

    for tag, token_value_pair in num_of_tags_and_tokens.items():
        for token, val in token_value_pair.items():
            value = q2aprob(val, num_of_tags[tag], len(wordList))
            output_file.write(token + " " + tag + " " + f'{value}' + "\n")

    for tag in num_of_tags:
        value = q2aprob(0, num_of_tags[tag], len(wordList))
        output_file.write("Na" + " " + tag + " " + f'{value}' + "\n")    

    train_file.close()
    tag_file.close()
    output_file.close()

def q2aprob(countxy, totaly, num_of_word, delta = 1):
    return round((countxy + delta)/(totaly + delta * (num_of_word + 1)), 25)

# Question 2b
q2a()

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    output_probs = pd.read_csv(in_output_probs_filename, delimiter=" ")

    test_file = open(in_test_filename, 'r', encoding='utf8')
    data = test_file.readlines()
    naive_prediction_file = open(out_prediction_filename, 'w', encoding='utf8')
    
    for tweet in data:
        tweet = tweet.split()
        for token in tweet:
            df = output_probs[output_probs['Token'] == token.upper()]
            tag = ""
            if df.empty:
                df = output_probs[output_probs['Token'] == 'Na']
                if token[0] == "@":
                    tag = "@"
                elif token[0] == "#":
                    tag = "#"
                elif token[:4] == "HTTP" or token[:3] == "WWW":
                    tag = "U"
                elif token[0].isnumeric():
                    tag = "$"
            df = df.sort_values(by=['Prob'], ascending=False)
            if tag == "":
                tag = df['Tag'].iloc[0]

            naive_prediction_file.write(tag) 
        naive_prediction_file.write('\n')
    test_file.close()
    naive_prediction_file.close()

# Question 2c
""" The accuracy of the predictions is 1017/1378 = 0.738 (3s.f) """

# Question 3a
""" 
P(y=j | x=w) = P(y=j, x=w) / P(x=w)
P(x = w) is constant. So we will want to obtain P(y=j, x=w). 
P(y=j, x=w) = P(x=w | y=j) x P(y=j)

P(x=w | y=j) is the value in naive_output_probs.txt. 
P(y=j) is the probability of the number of times tag j appears
P(x=w) is the number of times token w appears
"""

# Question 3b
def q3prob(train_file_name):
    train_file = open(train_file_name, 'r', encoding='utf8')
    train_data = train_file.readlines()
    tag_dict = dict()

    for line in train_data:
        token_tag_pair = line.split()
        if len(token_tag_pair) == 0:
            continue
        tag = token_tag_pair[1]

        tag_dict[tag] = tag_dict.get(tag, 0) + 1

    totaltags = sum(tag_dict.values())

    for tag in tag_dict:
        tag_dict[tag] /= totaltags

    
    train_file.close()
    return tag_dict

def q3_get_token_dict(test_data):
    output = {}
    for tweet in test_data:
        tweet = tweet.split()
        for token in tweet:
            output[token] = output.get(token, 0) + 1
    
    totaltokens = sum(output.values())

    for token in output:
        output[token] /= totaltokens

    return output

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    # in_output_probs_filename = "naive_output_probs.txt"
    # in_train_filename = "twitter_train.txt"
    # in_test_filename = "twitter_dev_no_tag.txt"
    # out_prediction_filename = "naive_predictions2.txt"

    output_probs = pd.read_csv(in_output_probs_filename, delimiter=" ")
    test_file = open(in_test_filename, 'r', encoding='utf8')
    test_data = test_file.readlines()

    tagsprob = q3prob(in_train_filename) # Dictionary of P(y = j)

    naive_pred2 = open(out_prediction_filename, 'w', encoding='utf8')

    for tweet in test_data:
        tweet = tweet.split()
        for token in tweet:
            df = output_probs[output_probs['Token'] == token.upper()]
            tag = ""
            if df.empty:
                df = output_probs[output_probs['Token'] == 'Na']
                if token[0] == "@":
                    tag = "@"
                elif token[0] == "#":
                    tag = "#"
                elif token[:4] == "HTTP" or token[:3] == "WWW":
                    tag = "U"
                elif token[0].isnumeric():
                    tag = "$"
            # After sorting, we will only get the tags for the token we want
            
            for i, row in df.iterrows():
                tag_temp = row['Tag']
                df.at[i, 'Prob'] *= tagsprob[tag_temp]
            df = df.sort_values(by=['Prob'], ascending=False)
            if tag == "":
                tag = df['Tag'].iloc[0]

            naive_pred2.write(tag) 
        naive_pred2.write('\n')
    test_file.close()
    naive_pred2.close()

# Question 3c
""" The accuracy of the prediction2 is 1057/1378 = 0.767 (3s.f) """

# Question 4a
# Getting output_probs
def get_output_probs():
    output_file_name ='output_probs.txt'
    train_file = open('twitter_train.txt', 'r', encoding='utf8')
    tag_file = open('twitter_tags.txt', 'r', encoding='utf8')
    output_file = open(output_file_name, 'w', encoding='utf8')

    train_lines = train_file.readlines()
    tag_lines = tag_file.readlines()

    num_of_tags = dict() # {"@": 10}
    num_of_tags_and_tokens  = dict()

    wordList = set()

    output_file.write("Token Tag Prob \n")

    for line in tag_lines:
        tag = line.split()[0]
        num_of_tags[tag] = 0
        num_of_tags_and_tokens[tag] = {}

    for line in train_lines:
        token_tag_pair = line.split()
        if len(token_tag_pair) == 0:
            continue
        token = token_tag_pair[0].upper()
        tag = token_tag_pair[1]

        num_of_tags[tag] += 1

        if token not in num_of_tags_and_tokens[tag]:
            num_of_tags_and_tokens[tag][token] = 1
        else:
            num_of_tags_and_tokens[tag][token] += 1

        wordList.add(token)

    for tag, token_value_pair in num_of_tags_and_tokens.items():
        for token, val in token_value_pair.items():
            value = q2aprob(val, num_of_tags[tag], len(wordList))
            output_file.write(token + " " + tag + " " + f'{value}' + "\n")

    for tag in num_of_tags:
        value = q2aprob(0, num_of_tags[tag], len(wordList))
        output_file.write("Na" + " " + tag + " " + f'{value}' + "\n")    

    train_file.close()
    tag_file.close()
    output_file.close()

def q2aprob(countxy, totaly, num_of_word, delta = 0.01):
    return round((countxy + delta)/(totaly + delta * (num_of_word + 1)), 25)

# Getting trans_probs
def populate_transition_probabilities():
    trans_file_name= 'trans_probs.txt'

    train_file = open('twitter_train.txt', 'r', encoding='utf8')
    tag_file = open('twitter_tags.txt', 'r', encoding='utf8')
    trans_file = open(trans_file_name, 'w', encoding='utf8')

    train_lines = train_file.readlines()
    tag_lines = tag_file.readlines() 
    trans_file.write("i j Prob \n")

    num_tag_dict= {}
    tag_dict = {}
    tag_dict_idx = {}
    tag_dict["START"] = (0, "START")
    tag_dict_idx[0] = "START"
    idx = 1

    for line in tag_lines:
        tag = line.split()[0]
        tag_dict_idx[idx] = tag
        tag_dict[tag] = (idx, tag)
        num_tag_dict[tag] = 0
        idx += 1

    tag_dict["STOP"] = (idx, "STOP")
    tag_dict_idx[idx] = "STOP"
    len_dict = len(tag_dict)
    # Creates a table that tabulates the different arrangements from i to j
    adj_matrix = np.zeros((len_dict, len_dict)) 

    i = "START"
    for line in train_lines:
        token_tag_pair = line.split()
        # token = token_tag_pair[0]
        # tag = token_tag_pair[1]

        if i == "START":
            num_tag_dict["START"] = num_tag_dict.get("START", 0) + 1

        if len(token_tag_pair) == 0:
            i_index = tag_dict[i][0]
            j_index = tag_dict["STOP"][0]
            adj_matrix[i_index][j_index] += 1
            i = "START"
            num_tag_dict["STOP"] = num_tag_dict.get("STOP", 0) + 1
        else:
            tag = token_tag_pair[1]
            i_index = tag_dict[i][0]
            j_index = tag_dict[tag][0]
            adj_matrix[i_index][j_index] += 1
            num_tag_dict[tag] += 1
            i = tag

    delta = 0.01

    for index in range(0, len_dict - 1):
        # row_array gives the i-th row
        row_array = adj_matrix[index]

        # tag_i is the i-th tag
        tag_i = tag_dict_idx[index]
        for tag_index in range(1, len_dict):
            tag_j = tag_dict_idx[tag_index]

            value = (row_array[tag_index] + delta) / (num_tag_dict[tag_i] + delta * (len_dict - 2))
            trans_file.write(tag_i + " " + tag_j + " " + f'{value}' + '\n')

    train_file.close()
    tag_file.close()
    trans_file.close()

# Question 4b
def parse_data(file):
    file = open(file, 'r', encoding= 'utf8')
    lines = file.readlines()
    tweet = []
    all_tweet = []
    for line in lines:
        line = line.split()
        if (len(line) == 0):
            all_tweet.append(tweet)
            tweet = []
        else:
            tweet.append(line[0].upper())
    if tweet:
        all_tweet.append(tweet)
    file.close()
    return all_tweet

def trans_prob_q4(df):
    trans_dict = {}
    for index, row in df.iterrows():
        trans_dict[(row["i"], row['j'])] = row['Prob']
    return trans_dict

def output_prob_q4(df):
    output_dict = {}
    for index, row in df.iterrows():
        output_dict[(row['Tag'], row["Token"])] = row["Prob"]
    return output_dict

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
    get_output_probs()
    populate_transition_probabilities()

    trans_probs = pd.read_csv(in_trans_probs_filename, delimiter=" ")
    output_probs = pd.read_csv(in_output_probs_filename, delimiter=" ")

    tag_file = open(in_tags_filename, 'r', encoding='utf8')
    tag_lines = tag_file.readlines()
    tags = {}
    tags_dict = {}
    idx = 0

    for tag in tag_lines:
        tag = tag.split()[0]
        tags_dict[tag] = idx
        tags[idx] = tag
        
        idx += 1

    data = parse_data(in_test_filename)
    trans_dict = trans_prob_q4(trans_probs)
    output_dict = output_prob_q4(output_probs)
    output_prediction = open(out_predictions_filename, 'w', encoding='utf8')

    for tweet in data:
        num_word = len(tweet)
        num_tag = len(tags)
        pi = np.zeros((num_tag, num_word))
        bp = np.zeros((num_tag, num_word), dtype=object)
        first_word = tweet[0]

        #start state
        for i in range(num_tag):
            transition = trans_dict[("START", tags[i])]
            output = output_dict.get((tags[i],first_word), output_dict[(tags[i], "Na")])
            pi[i][0] = transition * output
            bp[i][0] = "START"

        # recursive state
        for k in range(1, num_word):
            for v in range(num_tag):
                max_prob = -1
                for u in range(num_tag): # pprevious tag
                    transition_uv = trans_dict[(tags[u], tags[v])]
                    output_v = output_dict.get((tags[v],tweet[k]), output_dict[(tags[v], "Na")])
                    pi_prev = pi[u][k-1]

                    temp_prob = pi_prev * transition_uv * output_v
                    if temp_prob > max_prob:
                        max_prob = temp_prob
                        max_tag = tags[u]
                pi[v][k] = max_prob
                bp[v][k] = max_tag

        #stop state
        max_prob = -1
        max_tag = ''
        for v in range(num_tag):
                transition_vstop = trans_dict[(tags[v], "STOP")]
                pi_prev = pi[v][num_word-1]
                temp_prob = transition_vstop * pi_prev
                if temp_prob > max_prob:
                    max_prob = temp_prob
                    max_tag = tags[v]

        tag_list = []
        tag_list.append(max_tag)

        for i in range(num_word-1, 0, -1):
            possible_tag = bp[:,i]
            max_tag = possible_tag[tags_dict[max_tag]]
            tag_list.append(max_tag)

        for tag in tag_list[::-1]:
            output_prediction.write(tag + "\n")
        output_prediction.write('\n')

    tag_file.close()
    output_prediction.close()

# Question 4c
"""
Viterbi prediction accuracy:   1058/1378 = 0.7677793904208998
"""

# Question 5a
"""
For the preprocessing of the data, we will first be identifying specific patterns attributed to the token and assign a tag to it respectively. For example, if the token starts with "@USER_", then we will assign this a tag of @. We also added the ability to identify common emoticons. For example, if the token starts with ":)", then we will assign this a tag of E. Hence, upon reading any test data, any token that conforms to certain patterns will be assigned a tag as such. With this, we will be able to more accurately assign tags. We believe that this is an appropriate approach as we observed that many tokens with the same conventions are able to be classified as the same tag, preventing Viterbi from misclassifying.

Next, we will take into account linguistic patterns such as the Subject-Verb-Object word order where both tags from t-1 and t-2 should have an effect on tag t. A few examples of this would be: "Thank you! :)", where tags assigned to "Thank" and "you" should have an impact on ":)" instead of simply looking at "you" to decide the tag of ":)". In Question 4, the Viterbi algorithm uses a first-order Hidden Markov Model, in other words, a bigram tagger. Hence, we would like to improve this by implementing a second-order Hidden Markov Model, in other words, a trigram tagger. This is commonly used within the best statistical taggers (Thede & Harper, 1999).

Hence, our Viterbi Algorithm will be as such:
Start State: P(Y_1 | Y_0) * P(x_1 | Y_1) * P(Y_0) = pi(1, Y_1)

Recursive State: max_{y_{k-1}, y_{k-2}} pi(k-1, Y_{k-1}) * P(Y_k | Y_{k-1}, Y_{k-2}) * P(x_k | Y_k)

Stop State: P(Y_{n+1} | Y_n, Y_{n-1}) * pi(k-1, Y_{k-1})
"""

# Question 5b
def get_output_probs_2():
    output_file_name ='output_probs2.txt'
    train_file = open('twitter_train.txt', 'r', encoding='utf8')
    tag_file = open('twitter_tags.txt', 'r', encoding='utf8')
    output_file = open(output_file_name, 'w', encoding='utf8')

    train_lines = train_file.readlines()
    tag_lines = tag_file.readlines()

    num_of_tags = dict() # {"@": 10}
    num_of_tags_and_tokens  = dict()

    l = set()
    ll = set()
    wordList = set()
    output_file.write("Token Tag Prob \n")

    for line in tag_lines:
        tag = line.split()[0]
        num_of_tags[tag] = 0
        num_of_tags_and_tokens[tag] = {}

    for line in train_lines:
        token_tag_pair = line.split()
        if len(token_tag_pair) == 0:
            continue
        token = token_tag_pair[0].upper()
        tag = token_tag_pair[1]

        num_of_tags[tag] += 1
        if tag == ',':
            l.add(token)
        if token not in num_of_tags_and_tokens[tag]:
            num_of_tags_and_tokens[tag][token] = 1

        if tag == 'E' and not bool(re.search(r"^[a-zA-Z]+$", token)) and not bool(re.search(r'/d',token)):
            ll.add(token)
            #if any(char in emoji.UNICODE_EMOJI['en'] for char in token):
                #print(token)
        else:
            num_of_tags_and_tokens[tag][token] += 1

        wordList.add(token)

    for tag, token_value_pair in num_of_tags_and_tokens.items():
        for token, val in token_value_pair.items():
            value = q2aprob(val, num_of_tags[tag], len(wordList))
            output_file.write(token + " " + tag + " " + f'{value}' + "\n")

    for tag in num_of_tags:
        value = q2aprob(0, num_of_tags[tag], len(wordList))
        output_file.write("Na" + " " + tag + " " + f'{value}' + "\n")    
    
    train_file.close()
    tag_file.close()
    output_file.close()
    return l, ll

get_output_probs_2()

def populate_new_trans_prob():
    trans_file_name= 'trans_probs2.txt'
    train_file = open('twitter_train.txt', 'r', encoding='utf8')
    tag_file = open('twitter_tags.txt', 'r', encoding='utf8')
    trans_file = open(trans_file_name, 'w', encoding='utf8')

    
    train_lines = train_file.readlines()
    tag_lines = tag_file.readlines() 
    trans_file.write("i j Prob \n")

    num_tag_dict= {}
    tag_dict = {}
    tag_dict_idx = {}
    tag_dict["START"] = (0, "START")
    tag_dict_idx[0] = "START"
    ij_dict1 = {}
    idx = 1

    # Generate Dictionary for (ij, k) counts
    for line in tag_lines:
        tag_i = line.split()[0]
        tag_dict_idx[idx] = tag_i
        tag_dict[tag_i] = (idx, tag_i)
        num_tag_dict[tag_i] = 0
        idx0 = 0

        for line0 in tag_lines:
            tag_j = line0.split()[0]
            idx0 += 1
            if (tag_i + " " + tag_j) not in ij_dict1:
                ij_dict1[tag_i + " " + tag_j] = {}

            for line1 in tag_lines:
                tag_k = line1.split()[0]
                ij_dict1[tag_i + " " + tag_j][tag_k] = 0
            ij_dict1[tag_i + " " + tag_j]["STOP"] = 0
        idx += 1

    for line in tag_lines:
        tag = line.split()[0]
        ij_dict1["START" + " " + tag]= {}

        for line1 in tag_lines:
            tag1 = line1.split()[0]
            ij_dict1["START" + " "+ tag][tag1] = 0
        ij_dict1["START" + " " +tag]["STOP"] = 0
    
    tag_dict["STOP"] = (idx, "STOP")
    tag_dict_idx[idx] = "STOP"
    len_dict = len(tag_dict)
    adj_matrix = np.zeros((len_dict, len_dict))
    i = "START"

    for line in train_lines:
        token_tag_pair = line.split()
        if len(token_tag_pair) == 0:
            i_index = tag_dict[i][0]
            j_index = tag_dict["STOP"][0]
            adj_matrix[i_index][j_index] += 1
            i = "START"
        else:
            tag = token_tag_pair[1]
            i_index = tag_dict[i][0]
            j_index = tag_dict[tag][0]
            adj_matrix[i_index][j_index] += 1
            num_tag_dict[tag] += 1
            i = tag
    num_start = adj_matrix[0].sum(axis=0)

    if (i != "STOP"):
            i_index = tag_dict[i][0]
            j_index = tag_dict["STOP"][0]
            adj_matrix[i_index][j_index] += 1   
    num_stop = adj_matrix[:,-1].sum(axis=0)
    num_tag_dict["START"] = num_start
    num_tag_dict["STOP"] = num_stop
    start_array = adj_matrix[0]

    delta = 1

    # Populate (START, j) for trans_probs2 which utilises smoothing that is similar to that of a smoothing in Q4.
    for j in tag_dict.keys():
        if j == "START" or j == "STOP":
            continue
        index_j = tag_dict[j][0]
        value = (start_array[index_j] + delta) / (num_tag_dict["START"] + delta*(len_dict - 2))
        trans_file.write("START" + " " + j + " " + f'{value}' + '\n')

    start = True
    start_count = 0
    time2_list = []

    for line in train_lines:
        token_tag_pair = line.split()
        if len(token_tag_pair) == 0:
            start = True
            tag_i = time2_list[0]
            tag_j = time2_list[-1]
            ij_dict1[tag_i + " " + tag_j]["STOP"] += 1
            start_count = 0
            time2_list = []
            continue

        if start:
            tag_k = token_tag_pair[1]
            time2_list.append(tag_k)
            start_count += 1
            start = False
            continue

        if start_count == 1:
            tag_k = token_tag_pair[1]
            time2_list.append(tag_k)
            start_count += 1
            continue
        else:
            tag_i = time2_list[0]
            tag_j = time2_list[1]
            tag_k = token_tag_pair[1]
            ij_dict1[tag_i + ' '+ tag_j][tag_k]+= 1
            time2_list[0] = tag_j
            time2_list[1] = tag_k

    for key, item in ij_dict1.items():
        # total is the number of i,j tags
        total = 0
        for key1 in item:
            total += item[key1]
        for key1 in item:
            # val is the number of k tag given i, j tags
            val = item[key1]
            value = (val + delta) / (total + delta * (len(ij_dict1)))
            key2 = key.split()
            
            trans_file.write(key2[0]+','+key2[1] + " " + key1 + " " + f'{value}' + '\n')

    train_file.close()
    tag_file.close()
    trans_file.close()

populate_new_trans_prob()

def trans_prob_q5(df):
    trans_dict = {}
    trans_dict2 = {}
    for index, row in df.iterrows():
        if len(row['i']) == 1 or row['i'] == "START":
            trans_dict[(row["i"], row['j'])] = row['Prob']
        else:
            trans_dict2[(row['i']), row['j']] = row['Prob']
    return trans_dict, trans_dict2

def output_prob_q5(df):
    output_dict = {}
    for index, row in df.iterrows():
        output_dict[(row['Tag'], row["Token"])] = row["Prob"]
    return output_dict

def handle_regex(word, l, ll):
    checker = False

    if word[0:6] == "@USER_":
        word = "@"
        checker = True
        
    elif word[:4] == "HTTP" or word[:3] == "WWW":
        word= "U"
        checker = True

    elif word[0].isnumeric():
        word = "$"
        checker = True
        
    elif bool(re.fullmatch(r'^[-+]?[0-9,]*\.?[0-9]+([eE][-+]?[0-9]+)?$', word)):
        word = "$"
        checker = True

    elif bool(re.fullmatch(r'(:-?(:|\)|\[|\]|\\|\/|#|\$|3)|;-\(|:-?D|D-:|:-?P|:-?\/|:-?\(|;-?\(|:-?\||-_-|:-?\[|:-?\]|:-?O|:-?S|:-?\\|:-?@|:-?\$|:-?\'\(|:-?\'\)|（＾ｖ＾）|（＾ｕ＾）|（＾◇＾）|（＾＾）|（\?\*\^_\^\?\*\)|\(^_-\)|\\\(^o^\)\/|（〜\^∇\^）〜|o\(^▽^\)o|d\[-_-\]b)', word)):
        word = "E"
        checker = True

    elif word in l:
        word = ","
        checker = True

    elif word in ll:
        word = "E"
        checker = True

    # elif '#' in word:
    #     if 'HTTP://' not in word or 'HTTPS://' not in word:
    #         word = "#"
    #         checker = True
    
    #     elif word != '^':
    #         word = "#"
    #         checker = True
    
    elif word[0] == "#":
        word = "#"
        checker = True
    
    
    
    if checker:
        return {"checker": True, "new_tag": word}
    else:
        return {"checker": False}

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    trans_probs = pd.read_csv(in_trans_probs_filename, delimiter=" ")
    output_probs = pd.read_csv(in_output_probs_filename, delimiter=" ")
    tag_file = open(in_tags_filename, 'r', encoding='utf8')

    tag_lines = tag_file.readlines()
    tags = []
    tags_dict = {}
    idx = 0
    for tag in tag_lines:
        tag = tag.split()[0]
        tags_dict[tag] = idx
        tags.append(tag)
        
        idx += 1

    data = parse_data(in_test_filename)
    trans_dict, trans_dict2 = trans_prob_q5(trans_probs)
    output_dict = output_prob_q5(output_probs)
    output_prediction = open(out_predictions_filename, 'w', encoding='utf8')
    l, ll = get_output_probs_2()

    for tweet in data:
        num_word = len(tweet)
        num_tag = len(tags)
        pi = np.zeros((num_tag, num_word))
        bp = np.zeros((num_tag, num_word), dtype=object)

        first_word = tweet[0]
        first = first_word.split()

        for i in range(num_tag):
            transition = trans_dict[("START", tags[i])]
            output = output_dict.get((tags[i],first_word), output_dict[(tags[i], "Na")])
            pi[i][0] = transition * output
            bp[i][0] = "START"

        regex_checker = handle_regex(first[0], l, ll)

        if (regex_checker["checker"]):
            new_tag = regex_checker["new_tag"]
            k = tags_dict[new_tag]
            for i in range(0, num_tag):
                if i == k:
                    pi[i][0] = 1
                else:
                    pi[i][0] = 0

        # recursive state
        for k in range(1, num_word):
            for v in range(num_tag):
                max_prob = -1
                if k ==1:
                    for u in range(num_tag):
                        previous2 = "START" + "," + tags[u]
                        output_v = output_dict.get((tags[v],tweet[k]), output_dict[(tags[v], "Na")])
                        pi_prev = pi[u][k-1]               
                        transition_uuv = trans_dict2[(previous2, tags[v])]
                        temp_prob = pi_prev * output_v * transition_uuv
                        if temp_prob > max_prob:
                            max_prob = temp_prob
                            max_tag = tags[u]
                    pi[v][k] = max_prob
                    bp[v][k] = max_tag
                else: 
                    for u in range(num_tag): # previous tag
                        for pre in range(num_tag):
                            output_v = output_dict.get((tags[v],tweet[k]), output_dict[(tags[v], "Na")])
                            pi_prev = pi[u][k-1]
                            previous2 = tags[pre]+"," + tags[u] 
                            transition_uuv = trans_dict2[(previous2, tags[v])]

                            temp_prob = pi_prev * output_v * transition_uuv
                            if temp_prob > max_prob:
                                max_prob = temp_prob
                                max_tag = tags[u] 
                                prev_max_tag = tags[pre]   
                        pi[v][k] = max_prob
                        bp[v][k] = max_tag

            regex_checker = handle_regex(tweet[k], l, ll)
            if (regex_checker["checker"]):
                new_tag = regex_checker["new_tag"]
                m = tags_dict[new_tag]
                for i in range(0, num_tag):
                    if i == m:
                        pi[i][k] = 1
                    else:
                        pi[i][k] = 0
        #stop state
        max_prob = -1
        max_tag = ''
        for v in range(num_tag):
                if num_word == 1:
                    pi_prev = pi[v][num_word-1]
                    previous2 = bp[v][-1] +"," + tags[v]
                    transition_uuv = trans_dict2[(previous2, "STOP")]
                    temp_prob = pi_prev * transition_uuv
                    if temp_prob > max_prob:
                        max_prob = temp_prob
                        max_tag = tags[v]
                else:
                    for pre in range(num_tag):
                            pi_prev = pi[v][num_word-1]
                            previous2 = tags[pre]+"," + tags[v] 
                            transition_uuv = trans_dict2[(previous2, "STOP")]
                            temp_prob = pi_prev * transition_uuv
                            if temp_prob > max_prob:
                                max_prob = temp_prob
                                max_tag = tags[v] 
                                prev_max_tag = tags[pre]   

        tag_list = []
        tag_list.append(max_tag)
        for i in range(num_word-1, 0, -1):
            possible_tag = bp[:,i]
            max_tag = possible_tag[tags_dict[max_tag]]
            tag_list.append(max_tag)
        m=0

        for tag in tag_list[::-1]:
            output_prediction.write(tag + "\n")
        output_prediction.write('\n')

    tag_file.close()
    output_prediction.close()

# Question 5c
"""
Viterbi2 prediction accuracy:   1139/1378 = 0.8265602322206096
"""

# References
"""
Thede, S. M., & Harper, M. P. (1999). A Second-Order Hidden Markov Model for Part-of-Speech Tagging. ACL Anthology. Retrieved April 14, 2023, from https://aclanthology.org/P99-1023.pdf
"""

def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = 'C:/Users/yujie/Desktop/NUS/Y2S2/BT3102/Project/projectfiles' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    # naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'

    # naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    # naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    # correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    # print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    # naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    # naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    # correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    # print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                     viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    run()
