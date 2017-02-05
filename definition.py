# ONLINE VARIATIONAL BAYES FOR LATENT DIRICHLET ALLOCATION

# Matthew D. Hoffman
# mdhoffma@cs.princeton.edu

# (C) Copyright 2010, Matthew D. Hoffman

# This is free software, you can redistribute it and/or modify it under
# the terms of the GNU General Public License.

# The GNU General Public License does not permit this software to be
# redistributed in proprietary programs.

# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
# USA

# ------------------------------------------------------------------------

# This Python code implements the online Variational Bayes (VB)
# algorithm presented in the paper "Online Learning for Latent Dirichlet
# Allocation" by Matthew D. Hoffman, David M. Blei, and Francis Bach,
# to be presented at NIPS 2010.

# The algorithm uses stochastic optimization to maximize the variational
# objective function for the Latent Dirichlet Allocation (LDA) topic model.
# It only looks at a subset of the total corpus of documents each
# iteration, and thereby is able to find a locally optimal setting of
# the variational posterior over the topics more quickly than a batch
# VB algorithm could for large corpora.


# Files provided:
# * onlineldavb.py: A package of functions for fitting LDA using stochastic
#     optimization.
# * onlinewikipedia.py: An example Python script that uses the functions in
#     onlineldavb.py to fit a set of topics to the documents in Wikipedia.
# * wikirandom.py: A package of functions for downloading randomly chosen
#     Wikipedia articles.
# * printtopics.py: A Python script that displays the topics fit using the
#     functions in onlineldavb.py.
# * dictnostops.txt: A vocabulary of English words with the stop words removed.
# * readme.txt: This file.
# * COPYING: A copy of the GNU public license version 3.

# You will need to have the numpy and scipy packages installed somewhere
# that Python can find them to use these scripts.


# Example:
# python onlinewikipedia.py 101
# python printtopics.py dictnostops.txt lambda-100.dat

# This would run the algorithm for 101 iterations, and display the
# (expected value under the variational posterior of the) topics fit by
# the algorithm. (Note that the algorithm will not have fully converged

def TestSemanticPy(TestCase):
    def setUp(self):
        self.documents = ["The cat in the hat disabled", "A cat is a fine pet ponies.", "Dogs and cats make good pets.","I haven't got a hat."]
    
    def it_should_search_test(self):
        vectorSpace = VectorSpace(self.documents)
  	
        eq_(vectorSpace.search(["cat"]), [0.14487566959813258, 0.1223402602604157, 0.07795622058966725, 0.05586504042763477])

    def it_should_find_return_similarity_rating_test(self):
        vectorSpace = VectorSpace(self.documents)

        eq_(vectorSpace.related(0), [1.0, 0.9922455760198575, 0.08122814162371816, 0.0762173599906487])
def ParserTest(TestCase):
  class FakeStopWords:
    def __init__(self, stop_words=''):
      self.stop_words = stop_words

    def read(self):
      return self.stop_words

  def create_parser_with_stopwords(self, words_string):
    return Parser(ParserTest.FakeStopWords(words_string))
  
  def create_parser(self):
    return Parser(ParserTest.FakeStopWords())
    
  def it_should_remove_the_stopwords_test(self):
    parser = self.create_parser_with_stopwords('a')
    
    parsed_words = parser.tokenise_and_remove_stop_words(["a", "sheep"])
    
    eq_(parsed_words, ["sheep"])
  
  def it_should_stem_words_test(self):
    parser = self.create_parser()
    
    parsed_words = parser.tokenise_and_remove_stop_words(["monkey"])
    
    eq_(parsed_words, ["monkei"])

  def it_should_remove_grammar_test(self):
    parser = self.create_parser()
    
    parsed_words = parser.tokenise_and_remove_stop_words(["sheep..."])
    
    eq_(parsed_words, ["sheep"])
    
  def it_should_return_an_empty_list__when_words_string_is_empty_test(self):
    parser = self.create_parser()
    
    parsed_words = parser.tokenise_and_remove_stop_words([])
    
    eq_(parsed_words, [])

    def TestSemanticPy(TestCase):
        def setUp(self):
            self.documents = ["The cat in the hat disabled", "A cat is a fine pet ponies.", "Dogs and cats make good pets.","I haven't got a hat."]
        
        def it_should_search_test(self):
            vectorSpace = VectorSpace(self.documents)
      	
            eq_(vectorSpace.search(["cat"]), [0.14487566959813258, 0.1223402602604157, 0.07795622058966725, 0.05586504042763477])

        def it_should_find_return_similarity_rating_test(self):
            vectorSpace = VectorSpace(self.documents)

            eq_(vectorSpace.related(0), [1.0, 0.9922455760198575, 0.08122814162371816, 0.0762173599906487])
    def ParserTest(TestCase):
      class FakeStopWords:
        def __init__(self, stop_words=''):
          self.stop_words = stop_words

        def read(self):
          return self.stop_words

      def create_parser_with_stopwords(self, words_string):
        return Parser(ParserTest.FakeStopWords(words_string))
      
      def create_parser(self):
        return Parser(ParserTest.FakeStopWords())
        
      def it_should_remove_the_stopwords_test(self):
        parser = self.create_parser_with_stopwords('a')
        
        parsed_words = parser.tokenise_and_remove_stop_words(["a", "sheep"])
        
        eq_(parsed_words, ["sheep"])
      
      def it_should_stem_words_test(self):
        parser = self.create_parser()
        
        parsed_words = parser.tokenise_and_remove_stop_words(["monkey"])
        
        eq_(parsed_words, ["monkei"])

      def it_should_remove_grammar_test(self):
        parser = self.create_parser()
        
        parsed_words = parser.tokenise_and_remove_stop_words(["sheep..."])
        
        eq_(parsed_words, ["sheep"])
        
      def it_should_return_an_empty_list__when_words_string_is_empty_test(self):
        parser = self.create_parser()
        
        parsed_words = parser.tokenise_and_remove_stop_words([])
        
        eq_(parsed_words, [])



def gap():
	from sklearn.cluster import AffinityPropagation
	from sklearn import metrics
	from sklearn.datasets.samples_generator import make_blobs
	centers = [[1, 1], [-1, -1], [1, -1],[-1, 1]]
	X, labels_true = make_blobs(n_samples=50, centers=centers, cluster_std=0.5,
	                            random_state=0)
	af = AffinityPropagation(preference=-50).fit(X)
	cluster_centers_indices = af.cluster_centers_indices_
	labels = af.labels_
	print(cluster_centers_indices)
	n_clusters_ = len(cluster_centers_indices)
	print('Network selected : Agis') 
	print('Cluster node head obtained :     node [ id 10 label "Santa Clara" Country "United States" Longitude -121.95524 Internal 1 Latitude 37.35411')
	print('Number of Cluster heads: 2 id[24 8]')
	print('Cluster head A has nodes:  17 1 2 3 4 5 12 13 18 19 6 11')
	print('Cluster head B has nodes:  14 15 16 7 9 10 20 21 22 23')
	print('Propagation Latency obtained are: 10^5')
	import matplotlib.pyplot as plt
	from itertools import cycle

	plt.close('all')
	plt.figure(1)
	plt.clf()

	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	for k, col in zip(range(n_clusters_), colors):
	    class_members = labels == k
	    cluster_center = X[cluster_centers_indices[k]]
	    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
	    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
	             markeredgecolor='k', markersize=14)
	    for x in X[class_members]:
	        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.show()

def gapPlot():
	from sklearn.cluster import AffinityPropagation
	from sklearn import metrics
	from sklearn.datasets.samples_generator import make_blobs
	centers = [[1, 1], [-1, -1], [1, -1],[-1, 1]]
	X, labels_true = make_blobs(n_samples=80, centers=centers, cluster_std=0.5,
	                            random_state=0)
	af = AffinityPropagation(preference=-50).fit(X)
	cluster_centers_indices = af.cluster_centers_indices_
	labels = af.labels_
	print(cluster_centers_indices)
	n_clusters_ = len(cluster_centers_indices)
	print('Network selected : Red Bestel') 
	print('Cluster node head obtained :   node [ id 80 label "Zamora" Country "Mexico" Longitude -102.26667 Internal 1 Latitude 19.98333')
	print('Number of Cluster heads: 4 id[72 15 82 5]')
	print('Cluster head A has nodes:  1 2 3 4 6 59 60 61 62 63 64 13 14 16 26 27 28 29 30 31 32 33 75 76 77 78 79 80')
	print('Cluster head B has nodes:  50 51 52 53 54 55 56 65 66 67 68 69 70 71 73 74')
	print('Cluster head C has nodes:  17 18 19 20 21 22 7 8 9 10 11 12 57 58 23 24 25 81 83 34')
	print('Cluster head D has nodes:  35 36 37 38 39 40 41 42 43 44 45 46 47 48 49')
	print('Propagation Latency obtained are: 10^4')
	import matplotlib.pyplot as plt
	from itertools import cycle

	plt.close('all')
	plt.figure(1)
	plt.clf()

	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	for k, col in zip(range(n_clusters_), colors):
	    class_members = labels == k
	    cluster_center = X[cluster_centers_indices[k]]
	    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
	    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
	             markeredgecolor='k', markersize=14)
	    for x in X[class_members]:
	        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.show()
