from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import gensim
import sys
import logging

#Will hold lists of test sets per analogy task
capital_world, currency, city_in_state, family, adjective_to_adverb, opposite, comparative, nationality_adjective = [], [], [], [], [], [], [], []
#Will hold accuracies of GloVe embedding per analogy task
glove_capital_world, glove_currency, glove_city_in_state, glove_family, glove_adjective_to_adverb, glove_opposite, glove_comparative, glove_nationality_adjective= 0, 0, 0, 0, 0, 0, 0, 0
#Will hold accuracies of LexVec embedding per analogy task
lexvec_capital_world, lexvec_currency, lexvec_city_in_state, lexvec_family, lexvec_adjective_to_adverb, lexvec_opposite, lexvec_comparative, lexvec_nationality_adjective= 0, 0, 0, 0, 0, 0, 0, 0

#Parsing word-text.v1.txt and storing text in corresponding arrays
with open('word-test.v1.txt') as file:
    for line in file:
        if(': capital-world' in line):
            line = next(file)
            while(':' not in line): #Continues until next analogy group is reached
                line = line.replace("\t", "")
                capital_world.append(line[:-1].lower()) #Appends line while removing the new line character '\n'
                line = next(file) #Goes to the next line
        if(': currency' in line):
            line = next(file)
            while (':' not in line): #Continues until next analogy group is reached
                line = line.replace("\t", "")
                currency.append(line[:-1].lower())  # Appends line while removing the new line character '\n'
                line = next(file) #Goes to the next line
        if(': city-in-state' in line):
            line = next(file)
            while (':' not in line): #Continues until next analogy group is reached
                line = line.replace("\t", "")
                city_in_state.append(line[:-1].lower())  # Appends line while removing the new line character '\n'
                line = next(file) #Goes to the next line
        if (': family' in line):
            line = next(file)
            while (':' not in line): #Continues until next analogy group is reached
                line = line.replace("\t", "")
                family.append(line[:-1].lower())  # Appends line while removing the new line character '\n'
                line = next(file) #Goes to the next line
        if (': gram1-adjective-to-adverb' in line):
            line = next(file)
            while (':' not in line): #Continues until next analogy group is reached
                line = line.replace("\t", "")
                adjective_to_adverb.append(line[:-1].lower())  # Appends line while removing the new line character '\n'
                line = next(file) #Goes to the next line
        if (': gram2-opposite' in line):
            line = next(file)
            while (':' not in line): #Continues until next analogy group is reached
                line = line.replace("\t", "")
                opposite.append(line[:-1].lower())  # Appends line while removing the new line character '\n'
                line = next(file) #Goes to the next line
        if (': gram3-comparative' in line):
            line = next(file)
            while (':' not in line): #Continues until next analogy group is reached
                line = line.replace("\t", "")
                comparative.append(line[:-1].lower())  # Appends line while removing the new line character '\n'
                line = next(file) #Goes to the next line
        if (': gram6-nationality-adjective' in line):
            line = next(file)
            while (':' not in line): #Continues until next analogy group is reached
                line = line.replace("\t", "")
                nationality_adjective.append(line[:-1].lower())  # Appends line while removing the new line character '\n'
                line = next(file) #Goes to the next line

######################################### 1st EMBEDDING (GloVe) #######################################################################
model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.50d.txt', binary = False) #Declare GloVe embedding model using gensim


################################## PROBLEM 2 (ANTONYMS IN SIMILAR WORDS LIST) #########################################################
print("TOP 10 MOST SIMILAR WORDS FROM GloVe MODEL FOR:") #All using cosine similarity
print("increase: ", model.wv.similar_by_word('increase', topn=10, restrict_vocab=None))
print("enter: ", model.wv.similar_by_word('enter', topn=10, restrict_vocab=None))
print("raise: ", model.wv.similar_by_word('raise', topn=10, restrict_vocab=None))
print("go: ", model.wv.similar_by_word('go', topn=10, restrict_vocab=None))
print("leave: ", model.wv.similar_by_word('leave', topn=10, restrict_vocab=None))
print()
#######################################################################################################################################



################################## PROBLEM 1 (1st Embedding Continued) ################################################################

print("GloVe EMBEDDING ACCURACIES ON ANALOGIES:")
for i in range(len(capital_world)):
    try:
        predictions = model.wv.most_similar(positive=[capital_world[i].split()[0], capital_world[i].split()[1]], negative=[capital_world[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if(capital_world[i].split()[3] == tempList[0]): #If the actual 4th word matches the predicted word
            glove_capital_world += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
glove_capital_world = float(glove_capital_world) / len(capital_world) #Gets accuracy by dividing total number right by total number possible
print("Glove capital_world accuracy: ", glove_capital_world)

for i in range(len(currency)):
    try:
        predictions = model.wv.most_similar(positive=[currency[i].split()[0], currency[i].split()[1]], negative=[currency[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (currency[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            glove_currency += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
glove_currency = float(glove_currency) / len(currency) #Gets accuracy by dividing total number right by total number possible
print("Glove currency accuracy: ", glove_currency)

for i in range(len(city_in_state)):
    try:
        predictions = model.wv.most_similar(positive=[city_in_state[i].split()[0], city_in_state[i].split()[1]], negative=[city_in_state[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (city_in_state[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            glove_city_in_state += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
glove_city_in_state = float(glove_city_in_state) / len(city_in_state) #Gets accuracy by dividing total number right by total number possible
print("Glove city_in_state accuracy: ", glove_city_in_state)

for i in range(len(family)):
    try:
        predictions = model.wv.most_similar(positive=[family[i].split()[0], family[i].split()[1]], negative=[family[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (family[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            glove_family += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
glove_family = float(glove_family) / len(family) #Gets accuracy by dividing total number right by total number possible
print("Glove family accuracy: ", glove_family)

for i in range(len(adjective_to_adverb)):
    try:
        predictions = model.wv.most_similar(positive=[adjective_to_adverb[i].split()[0], adjective_to_adverb[i].split()[1]], negative=[adjective_to_adverb[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (adjective_to_adverb[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            glove_adjective_to_adverb += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
glove_adjective_to_adverb = float(glove_adjective_to_adverb) / len(adjective_to_adverb) #Gets accuracy by dividing total number right by total number possible
print("Glove adjective_to_adverb accuracy: ", glove_adjective_to_adverb)

for i in range(len(opposite)):
    try:
        predictions = model.wv.most_similar(positive=[opposite[i].split()[0], opposite[i].split()[1]], negative=[opposite[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (opposite[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            glove_opposite += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
glove_opposite = float(glove_opposite) / len(opposite) #Gets accuracy by dividing total number right by total number possible
print("Glove opposite accuracy: ", glove_opposite)

for i in range(len(comparative)):
    try:
        predictions = model.wv.most_similar(positive=[comparative[i].split()[0], comparative[i].split()[1]], negative=[comparative[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (comparative[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            glove_comparative += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
glove_comparative = float(glove_comparative) / len(comparative) #Gets accuracy by dividing total number right by total number possible
print("Glove comparative accuracy: ", glove_comparative)

for i in range(len(nationality_adjective)):
    try:
        predictions = model.wv.most_similar(positive=[nationality_adjective[i].split()[0], nationality_adjective[i].split()[1]], negative=[nationality_adjective[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (nationality_adjective[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            glove_nationality_adjective += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
glove_nationality_adjective = float(glove_nationality_adjective) / len(nationality_adjective) #Gets accuracy by dividing total number right by total number possible
print("Glove nationality_adjective accuracy: ", glove_nationality_adjective)
print()





#####################################################################################################333################################
#####################################################################################################333################################
######################################### 2nd EMBEDDING (LEXVEC) #######################################################################

model2 = gensim.models.KeyedVectors.load_word2vec_format('lexvec.enwiki+newscrawl.300d.W.pos.vectors', binary = False) #Declare LexVec embedding model using gensim

print("LexVec EMBEDDING ACCURACIES ON ANALOGIES")
for i in range(len(capital_world)):
    try:
        predictions = model2.wv.most_similar(positive=[capital_world[i].split()[0], capital_world[i].split()[1]], negative=[capital_world[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (capital_world[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            lexvec_capital_world += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
lexvec_capital_world = float(lexvec_capital_world) / len(capital_world) #Gets accuracy by dividing total number right by total number possible
print("Lexvec capital_world accuracy: ", lexvec_capital_world)

for i in range(len(currency)):
    try:
        predictions = model2.wv.most_similar(positive=[currency[i].split()[0], currency[i].split()[1]], negative=[currency[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (currency[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            lexvec_currency += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
lexvec_currency = float(lexvec_currency) / len(currency) #Gets accuracy by dividing total number right by total number possible
print("Lexvec currency accuracy: ", lexvec_currency)

for i in range(len(city_in_state)):
    try:
        predictions = model2.wv.most_similar(positive=[city_in_state[i].split()[0], city_in_state[i].split()[1]], negative=[city_in_state[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (city_in_state[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            lexvec_city_in_state += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
lexvec_city_in_state = float(lexvec_city_in_state) / len(city_in_state) #Gets accuracy by dividing total number right by total number possible
print("Lexvec city_in_state accuracy: ", lexvec_city_in_state)

for i in range(len(family)):
    try:
        predictions = model2.wv.most_similar(positive=[family[i].split()[0], family[i].split()[1]], negative=[family[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (family[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            lexvec_family += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
lexvec_family = float(lexvec_family) / len(family) #Gets accuracy by dividing total number right by total number possible
print("Lexvec family accuracy: ", lexvec_family)

for i in range(len(adjective_to_adverb)):
    try:
        predictions = model2.wv.most_similar(positive=[adjective_to_adverb[i].split()[0], adjective_to_adverb[i].split()[1]], negative=[adjective_to_adverb[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (adjective_to_adverb[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            lexvec_adjective_to_adverb += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
lexvec_adjective_to_adverb = float(lexvec_adjective_to_adverb) / len(adjective_to_adverb) #Gets accuracy by dividing total number right by total number possible
print("Lexvec adjective_to_adverb accuracy: ", lexvec_adjective_to_adverb)

for i in range(len(opposite)):
    try:
        predictions = model2.wv.most_similar(positive=[opposite[i].split()[0], opposite[i].split()[1]], negative=[opposite[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (opposite[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            lexvec_opposite += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
lexvec_opposite = float(lexvec_opposite) / len(opposite) #Gets accuracy by dividing total number right by total number possible
print("Lexvec opposite accuracy: ", lexvec_opposite)

for i in range(len(comparative)):
    try:
        predictions = model2.wv.most_similar(positive=[comparative[i].split()[0], comparative[i].split()[1]], negative=[comparative[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (comparative[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            lexvec_comparative += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
lexvec_comparative = float(lexvec_comparative) / len(comparative) #Gets accuracy by dividing total number right by total number possible
print("Lexvec comparative accuracy: ", lexvec_comparative)

for i in range(len(nationality_adjective)):
    try:
        predictions = model2.wv.most_similar(positive=[nationality_adjective[i].split()[0], nationality_adjective[i].split()[1]], negative=[nationality_adjective[i].split()[2]]) #Gets the 10 most likely words that complete the analogy
        tempList = [x[0] for x in predictions] #Holds top 10 most likely words in descending order
        if (nationality_adjective[i].split()[3] == tempList[0]):  # If the actual 4th word matches the predicted word
            lexvec_nationality_adjective += 1
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
lexvec_nationality_adjective = float(lexvec_nationality_adjective) / len(nationality_adjective) #Gets accuracy by dividing total number right by total number possible
print("Lexvec nationality_adjective accuracy: ", lexvec_nationality_adjective)