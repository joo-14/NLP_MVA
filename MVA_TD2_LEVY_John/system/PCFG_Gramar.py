import numpy as np
import nltk
import nltk.tree as Tree
from nltk import treetransforms, induce_pcfg,  Nonterminal

class PCFG():
	""" 
	Our PCFG is given by gramar where terminals are part-of-speech tags
	The probabilistic lexicon i.e. triples of the form (token, part-of-speech tag, probability) such that the sum of the probabilities for all triples for a given token sums to 1.
	is given by lhs_terminal_occur
	All other dictionnary are useful either for OOV or PCYK

	"""
	def __init__(self):
		self.lexique_occur = {} #count the number of occurence for each word
		self.lhs_terminal_occur = {} #count the occurence for the lhs that leads to a terminal ( a word )
		self.lhs_occur = {} #count the occurence of lhs gramar rule 
		self.lhs_non_terminal_occur = {} #count the occurence for the lhs that leads to non terrminal ( grammar )
		self.lhs_terminal_proba = {} #the lhs with the proba associated when the rhs is a terminal ( a word )
		self.lhs_non_terminal_proba = {} #the lhs with the proba associated when the rhs is a terminal ( grammar )
		self.lexique = [] #all the words
		self.gramar = {} #This is the pcfg with in key the lhs and rhs and in value the proba
		self.count = {} #utils to get the proba for the pcfg
		self.dico_POS = {} #key are the lexicon and values are a list of all lhs rules that leads to this vocabulary word
		self.dico_POS_count = {} #utils to compute dico_POS_prob
		self.dico_POS_compteur = {} #utils to compute dico_POS_prob
		self.dico_POS_prob = {} #key are (part of speech,token) = occurence of (POS,token) / occurence of this POS



	def probability(self):

		for key in self.lhs_terminal_occur.keys():
			self.lhs_terminal_proba[key] = self.lhs_terminal_occur[key] / self.lexique_occur[key[1]]

		for key in self.lhs_non_terminal_occur.keys():
			self.lhs_non_terminal_proba[key] = self.lhs_non_terminal_occur[key] / self.lhs_occur[key[0]]


	def final_grammar(self):
		for key, value in self.gramar.items():
			self.gramar[key] /= self.count[key[0]]


	def creation_pcfg(self,tree): #create our pcfg

		essaie = tree.productions()
		for rule in essaie :
				if rule.is_lexical():
					self.gramar[(rule.lhs(),rule.rhs()[0])] = self.gramar.get((rule.lhs(),rule.rhs()),0) + 1 
					self.count[rule.lhs()] = self.count.get(rule.lhs(),0) + 1
				else : 
					self.gramar[(rule.lhs(),rule.rhs())] = self.gramar.get((rule.lhs(),rule.rhs()),0) + 1 
					self.count[rule.lhs()] = self.count.get(rule.lhs(),0) + 1



	def POS(self,tree): 

		for key in tree.productions():
			if key.is_lexical():
				self.dico_POS[key.rhs()[0]] = self.dico_POS.get(key.rhs()[0],[]) + [key.lhs()]
				self.dico_POS[key.rhs()[0]] = list(set(self.dico_POS[key.rhs()[0]])) # we remove doublons

				self.dico_POS_count[(key.rhs()[0],key.lhs())] = self.dico_POS_count.get((key.rhs()[0],key.lhs()),0) + 1
				self.dico_POS_compteur[key.lhs()] = self.dico_POS_compteur.get(key.lhs(),0) + 1
		
		for key in tree.productions():
			if key.is_lexical():
				self.dico_POS_prob[(key.lhs(),key.rhs()[0])] = self.dico_POS_count[(key.rhs()[0],key.lhs())] /self.dico_POS_compteur[key.lhs()]




	def creation_grammar(self,tree):

		for rule in tree.productions():

			if rule.is_lexical():
				
				lhs = rule.lhs()
				rhs = rule.rhs()[0]
				
				self.lexique_occur[rhs] = self.lexique_occur.get(rhs,0) + 1
				self.lhs_terminal_occur[(lhs,rhs)] = self.lhs_terminal_occur.get((lhs,rhs),0) + 1

				if lhs not in self.lhs_occur.keys():
					self.lhs_occur[lhs] = 0


			else :
				lhs = rule.lhs()
				rhs = rule.rhs()

				self.lhs_occur[lhs] = self.lhs_occur.get(lhs,0) + 1
				self.lhs_non_terminal_occur[(lhs,rhs)] = self.lhs_non_terminal_occur.get((lhs,rhs),0) + 1





	def chomsky(self) :

		with open("train_file","r") as txt :
			for i,sentence in enumerate(txt) :
				tree = nltk.tree.Tree.fromstring(sentence)
				self.lexique += [tree.leaves()[i] for i in range(len(tree.leaves()))]
				#we put into chomsky form
				tree.collapse_unary(collapsePOS = False)

				tree.chomsky_normal_form(horzMarkov = 2)

				self.creation_grammar(tree)

				self.creation_pcfg(tree)

				self.POS(tree)

				self.probability()


		self.final_grammar()


				






