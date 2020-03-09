import numpy as np
import nltk
import nltk.tree as Tree
from nltk import treetransforms, induce_pcfg,  Nonterminal
from copy import deepcopy
import pickle
import re


class OOV():
	def __init__(self):
		#lexique is the lexicon of our train
		#words et embeddinfgs are in our polyglot 
		#freq_lexique is our dictionnary where the keys are word from the train lexique and the values are the frequence of apparition
		#temp_dic is a dictionnary where the keys are wordof polyglot and the values are then embedding 
		#lexicon_embedding is a dico where (when it is possible) the keys are the trainig word corpus and the values the embedding
		#self.dico is a dictionnary where the keys are the words and the values are the index of the words
		#bigram_matrix is our language model
		self.parameter = 1e8
		self.words, self.embeddings = pickle.load(open('polyglot-fr.pkl', 'rb'),encoding = 'latin1')
		self.DIGITS = re.compile("[0-9]", re.UNICODE)
		self.lexique = []
		self.training_sequence = [] #we take all the sentence and put it in this tab
		self.sentence = []
		with open("train_file", "r") as txt:
			for phrase in txt:
			#on recupere le lexique
				self.sentence = []
				self.tree = nltk.tree.Tree.fromstring(phrase)
				self.lexique += [self.tree.leaves()[i] for i in range(len(self.tree.leaves()))]
				self.sentence += [self.tree.leaves()[i] for i in range(len(self.tree.leaves()))]
				self.training_sequence.append(self.sentence)


		self.freq_lexique = {}
		for word in self.lexique :
			if word not in self.freq_lexique.keys():
				self.freq_lexique[word] = 1
			else :
				self.freq_lexique[word] +=1

		self.temp_dic = {word: embedding for word, embedding in zip(self.words,self.embeddings)}

		self.lexicon_embedding = {}
		for word in self.lexique:
			if word in self.temp_dic.keys() :
				self.lexicon_embedding[word] = self.temp_dic[word]

		#we create the bigram matrix
		
		self.bigram_matrix = np.ones((len(self.freq_lexique),len(self.freq_lexique)))
		self.dico = {word: i for i,word in enumerate(self.freq_lexique.keys())}
		self.id = {i:word for i,word in enumerate(self.freq_lexique.keys())}
		
		for sentence in self.training_sequence:
			for i in range(len(sentence)-1):
				self.bigram_matrix[self.dico[sentence[i]],self.dico[sentence[i+1]]] +=1
		for i in range(len(self.freq_lexique)):
			if np.sum(self.bigram_matrix[i,:]) >0:
				self.bigram_matrix[i,:] /= np.sum(self.bigram_matrix[i,:])

  		
		

	def distance_Levenshtein(self,sentence1,sentence2):
	#initialization
		a,b = len(sentence1),len(sentence2)
		d = np.zeros((a+1,b+1))
		cost = 0

		for i in range(a+1):
			d[i,0] = i
		for j in range(b+1):
			d[0,j] = j

		for i in range(1,a+1):
			for j in range(1,b+1):
				if sentence1[i-1] == sentence2[j-1] :
					cost = 0
				else :
					cost = 1
				d[i,j] = min(d[i-1,j] + 1, #effacement nouveau caractere de sentence 1
					d[i,j-1] + 1, 	#insertion dans chaine 1 du nouveau caractere de sentence2
					d[i-1,j-1] + cost) #subsitution
					

		return d[a,b]


	def case_normalizer(self,word):
  	#In case the word is not available in the vocabulary,
 #    we can try multiple case normalizing procedure.
 #    We consider the best substitute to be the one with the lowest index,
 #    which is equivalent to the most frequent alternative.
  		w = word
  		lower = (self.freq_lexique.get(w.lower(), 1e12), w.lower())
  		upper = (self.freq_lexique.get(w.upper(), 1e12), w.upper())
  		title = (self.freq_lexique.get(w.title(), 1e12), w.title())
  		results = [lower, upper, title]
  		results.sort()
  		index, w = results[0]
  		if index != 1e12:
  			return w
  		return word


	def normalize(self,word):
		#Find the closest alternative in case the word is OOV.
		if not word in self.freq_lexique.keys():
			word = self.DIGITS.sub("#", word)
		if not word in self.freq_lexique.keys():
			word = self.case_normalizer(word)
		if not word in self.freq_lexique.keys(): #si on a rien trouvé on retourne le mot tel quel
			return None
		return word

	def probability_transition(self,w,previous_word,next_word):
		#this gives the transition based on our bigram matrix 

		#if the previous or the next word is not in our lexique we put None
		if previous_word not in self.freq_lexique.keys():
			previous_word = None
		if next_word not in self.freq_lexique.keys():
			next_word = None

		proba = 1
		if previous_word != None or next_word != None :
			if previous_word != None and next_word != None :
				proba = self.parameter*self.bigram_matrix[self.dico[previous_word],self.dico[w]]*self.bigram_matrix[self.dico[w],self.dico[next_word]]
			if previous_word == None :
				proba = 1e4*self.bigram_matrix[self.dico[w],self.dico[next_word]]
			if next_word == None :
				proba = 1e4*self.bigram_matrix[self.dico[previous_word],self.dico[w]]
		return proba

	def close_word_emb(self,word,previous_word,next_word):
	#we need to choose the closest word when word has an embedding, so the word exist in the corpus
		id_dico = {w:i for i,w in enumerate(self.lexicon_embedding.keys())}
		word_emb = self.temp_dic[word]
		candidat = []
		liste_word = [w for w in self.lexicon_embedding.keys()] #we take all the word in the lexicon_embedding dico
		#we compute the cos similarity for each candidate
		for w in self.lexicon_embedding.keys():
			cand_emb = self.lexicon_embedding[w] #embedding of the candidat
			similarity = np.dot(cand_emb,word_emb)/(np.linalg.norm(cand_emb)*np.linalg.norm(word_emb))
			candidat.append(similarity)
		potentiel = self.spelling_correction(word,True)

		
		if len(potentiel)==0:
			potentiel = self.lexicon_embedding.keys()
		#we add the language model by using the probability transition matrix, in this way we take into consideration the context 

		
		nouvelle_liste = []
		for i,w in enumerate(potentiel) :
			if w in self.lexicon_embedding.keys():
				nouvelle_liste.append(w)

		for w in nouvelle_liste :
			
			
			candidat[id_dico[w]] += self.probability_transition(w,previous_word,next_word)/(self.freq_lexique[w])
			
		indice = np.argmax(candidat)
		return liste_word[indice]

	#this function take care of all the mistake (3 max) and return the word
	def spelling_correction(self,word,boolen):
	#we return a list of candidates, the boolen is True when we are in embedding case, False otherwise. 
		liste_total = []
		liste1,liste2,liste3,liste4 = [],[],[],[]
		for w in self.freq_lexique.keys():
			dist = self.distance_Levenshtein(w,word)
			if dist == 1 :
				liste1.append(w)
			if dist == 2 :
				liste2.append(w)
			if dist == 3 :
				liste3.append(w)
			if dist == 4 :
				liste4.append(w)
			

		liste_total += liste1 + liste2 +liste3 + liste4 #we take just the two first list in case if we have an embedding
		

		if boolen : #if we are in closest word function so we search for word that has an embedding
			for w in liste_total :
				if w not in self.lexicon_embedding.keys():
					liste_total.remove(w)

			

		return liste_total

	def ponderation(self,word,w): #we put more weight when the levensthein distance is low
	#this function weight the candidates on the levensthein distance that this worrd has
		dist = self.distance_Levenshtein(w,word)
		#return (self.freq_lexique[w]/len(self.freq_lexique)) * 100**(1-dist)
		#return 1/(self.freq_lexique[w]/len(self.freq_lexique)) * 100**(1-dist)
		poids = 1
		if self.freq_lexique[w] >=200 :
			poids = self.freq_lexique[w]/len(self.freq_lexique)
		
		

		return 10**(1-dist) * poids

	def close_word_spe(self,word,previous_word,next_word):
		#in this function we put a weight depending on the levenstein distance because here we thing that the word is written with a fault
		#we do not need to put weight on the close_word_emb function because the word exisut it is just not the same as the other
		potentiel = self.spelling_correction(word,False)
		if len(potentiel) == 0: #if we have no candidates we take all the lexique 
			print("For word ", word, " we do not have nor close word in term of embedding, neither close word in term of spelling")
			potentiel = self.freq_lexique.keys()
		cand = np.zeros(len(self.freq_lexique))
		for w in potentiel : #next line : proba transition represents our language model and we add the candidat where the number of frequence are higher in the text
			cand[self.dico[w]] = self.probability_transition(w,previous_word,next_word) * self.ponderation(word,w)
		indice = np.argmax(cand)
		return self.id[indice]


	
	

	def verication_OOV(self,word,previous_word,next_word):


		if word in self.freq_lexique.keys():
			print("The word ", word," is in the lexique.")
			return word

		w = self.normalize(word)
		if w != None : #ca veut dire que normalize nous ressort None
			print("With normalize the word ", word, " is now ",w)
			return w 
		
		print("OOV word : ", word)
		if word in self.temp_dic.keys(): #we check if the word has an embedding
			print("The word ",word, " has an embedding")
			w = self.close_word_emb(word,previous_word,next_word)
			print("The closest word in term of embedding with language model is : ",w)
			return w

		w = self.close_word_spe(word,previous_word,next_word)
		print("The closest word in term of spelling with language model is : ",w)
		return w 





#test for our oov
"""

oov = OOV()

print("\n")
print(oov.verication_OOV("fiat","Il","que"))
print(oov.verication_OOV("non-",None,None))

print(oov.verication_OOV("assumer",None,None))

print(oov.verication_OOV("député-maire","le","-LRB-"))

print(oov.verication_OOV("Levallois-Perret","à","faire"))

print(oov.verication_OOV("37","dont","de"))
print("\n")


"""

