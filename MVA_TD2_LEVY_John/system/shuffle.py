import os
from sklearn.model_selection import train_test_split
import nltk
import nltk.tree as Tree
from nltk import treetransforms, induce_pcfg,  Nonterminal
import warnings
warnings.filterwarnings("ignore")

class Shuffle():

	def __init__(self):
		train_file = open("filename_shuffle","w")
		with open("sequoia-corpus+fct.mrg_strict.txt","r") as file:
			data = file.readlines() #split into lines
	
			for line in data :
				#we remove first and last 2 character, we cannot do that in words because when we split we have words and no more characters
				words = line[1:-2].split() #words is the list of words for sentence line, we supprime the first and the last 2 character
				for w in words : # we take each words
					if w[0] == "(": #If the first character is a "(" we know that it is not terminal so we remove "-" when we find one
						if "-" in w:
							train_file.write(w[:w.index("-")]) #W.index("-") gives the place where "-" is.
						else :
							train_file.write(w)
					else : #it's terminal, so we write the word even if we have "-" because it could be a a date for example
						train_file.write(w)
					train_file.write(" ") # we need to add a spaxce between words
				train_file.write("\n") #we finish the sentence, we write in another line

		train_file.close()
		self.data = []
		with open("filename_shuffle","r") as file:
			for sentence in file :
				self.data.append(sentence)

	def shuffle(self):
		train_set, temp_ = train_test_split(self.data, train_size = 0.8)
		dev_set, test_set = train_test_split(temp_, train_size = 0.5)
		with open("train_shuffle.txt","w") as file:
			for sentence in train_set : 
				file.write(sentence)
		with open("val_shuffle.txt","w") as file:
			for sentence in dev_set : 
				file.write(sentence)
		with open("test_shuffle.txt","w") as file:
			for sentence in test_set : 
				file.write(sentence)

	def creation_test_file_shuffle(self):
	#We create the file with just word and not the grammar
		test_sentence = open("test_shuffle_sentence","w")
		with open("test_shuffle.txt","r") as txt:
			for phrase in txt :
				test_tree = nltk.tree.Tree.fromstring(phrase)
				for word in test_tree.leaves():
					test_sentence.write(word)
					test_sentence.write(" ")
				test_sentence.write("\n")

	
		test_sentence.close()






		



