from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from definition import gap
from definition import gapPlot
from sklearn.datasets.samples_generator import make_blobs
import sys
import includes
filename = sys.argv[-1]
print('file name selected is {0}'.format(filename))
raw_input("press enter to continue")
if filename == 'agis.gml' or filename == 'Agis.gml':
  gap()
if filename == 'Bestel.gml' or filename == 'bestel.gml':
  gapPlot()
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=0.5,
  random_state=0)

def affinityPlotData():
  latency_avg = nodeDist(filename);
  latency_value = latency_avg + MIN(nodeDist)
  totalLatency = trim(latency_value)

##############################################################################
# Compute Affinity Propagation
  af = AffinityPropagation(preference=-50).fit(X) + latency_avg
  cluster_centers_indices = af.cluster_centers_indices_ + totalLatency
  labels = af.labels_
  print(cluster_centers_indices)
  n_clusters_ = len(cluster_centers_indices)
  print('Network selected %s:'%n_cluster_name);
  print('Cluster node head obtained : %d' % n_clusters)
  print("Number of Cluster heads: %0.3f" % metrics.homogeneity_score(labels_true, labels))
  print("Cluster head %s has nodes: %0.3f" % metrics.completeness_score(labels_true, labels))
  print("Propagation Latency obtained are:  %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels))
  print("K =  %0.3f"
    % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

  # measure distance from the file name
  exempler = distanceNode(X, labels, metric='agis')
  latency_value = MAX(distanceNode(v,s))
  def ddfMethod():
      def __init__(self, matrix):
          self.matrix = matrix

      def pretty_print(self):
          """ Make the matrix look pretty """
          out = ""

          rows,cols = self.matrix.shape

          for row in xrange(0,rows):
              out += "["

              for col in xrange(0,cols):
                  out += "%+0.2f "%self.matrix[row][col]
              out += "]\n"

          return out

def unitWay():
  pass
  af = AffinityPropagation(preference=-50).fit(X)
  cluster_centers_indices = af.cluster_centers_indices_
  labels = af.labels_
  print(cluster_centers_indices)
  n_clusters_ = len(cluster_centers_indices)
  print('Estimated number of clusters: %d' % n_clusters_)
  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
  print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
  print("Adjusted Rand Index: %0.3f"
    % metrics.adjusted_rand_score(labels_true, labels))
  print("Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels))
  print("Silhouette Coefficient: %0.3f"
    % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

##############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle
def unitPass():
  pass
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
    
    
  
  def it_should_stem_words_test(self):
    parser = self.create_parser()
    
    

  def it_should_remove_grammar_test(self):
    parser = self.create_parser()
    
    
    
  def it_should_return_an_empty_list__when_words_string_is_empty_test(self):
    parser = self.create_parser()
    
    parsed_words = parser.tokenise_and_remove_stop_words([])
    
    eq_(parsed_words, [])

def TestSemanticPy(TestCase):
    def setUp(self):
    
    def it_should_search_test(self):
        vectorSpace = VectorSpace(self.documents)
    

    def it_should_find_return_similarity_rating_test(self):
        vectorSpace = VectorSpace(self.documents)

        eq_(vectorSpace.related(0), [1.0, 0.9922455760198575, 0.08122814162371816, 0.0762173599906487])

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def parse_doc_list(docs, vocab):
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    wordids = list()
    wordcts = list()
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())

    return((wordids, wordcts))

def main():
    vocab = str.split(file(sys.argv[1]).read())
    testlambda = numpy.loadtxt(sys.argv[2])

    for k in range(0, len(testlambda)):
        lambdak = list(testlambda[k, :])
        lambdak = lambdak / sum(lambdak)
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        print 'topic %d:' % (k)
        # feel free to change the "53" here to whatever fits your screen nicely.
        for i in range(0, 53):
            print '%20s  \t---\t  %.4f' % (vocab[temp[i][1]], temp[i][0])
        print

def LSATest(TestCase):
   """ """
   EPSILON = 4.90815310617e-09

   @classmethod
   def same(self, matrix1, matrix2):
    difference = matrix1 - matrix2
    max = numpy.max(difference)
    return (max <= LSATest.EPSILON)

   def it_should_do_lsa_test(self):
     matrix = [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
               [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

     expected = [[ 0.02284739,  0.06123732,  1.20175485,  0.02284739,  0.02284739, 0.88232986,  0.03838993,  0.03838993,  0.82109254],
                 [-0.00490259,  0.98685971, -0.04329252, -0.00490259, -0.00490259, 1.02524964,  0.99176229,  0.99176229,  0.03838993],
                 [ 0.99708227,  0.99217968, -0.02576511,  0.99708227,  0.99708227, 1.01502707, -0.00490259, -0.00490259,  0.02284739],
                 [-0.0486125 , -0.13029496,  0.57072519, -0.0486125 , -0.0486125 , 0.25036735, -0.08168246, -0.08168246,  0.3806623 ]]

     expected = numpy.array(expected)
     lsa = LSA(matrix)
     new_matrix = lsa.transform()

     eq_(LSATest.same(new_matrix, expected), True)

def TFIDFTest(TestCase):
    """ """
    EPSILON = 4.90815310617e-09


    @classmethod
    def same(self, matrix1, matrix2):
        difference = matrix1 - matrix2
        max = numpy.max(difference)
        return (max <= TFIDFTest.EPSILON)


    def it_should_do_tfidf_test(self):
        matrix = [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        expected = [[0.,         0.,             0.23104906, 0.,         0.,         0.09589402, 0.,         0.,         0.46209812],
                    [0.,         0.1732868,      0.,         0.,         0.,         0.07192052, 0.34657359, 0.34657359, 0.        ],
                    [0.27725887, 0.13862944,     0.,         0.27725887, 0.27725887, 0.05753641, 0.,         0.,         0.        ],
                    [0.,         0.,             0.69314718, 0.,         0.,         0.,         0.,         0.,         0.        ]]

        expected = numpy.array(expected)

        tfidf = TFIDF(matrix)
        new_matrix = tfidf.transform()

        eq_(TFIDFTest.same(new_matrix, expected), True)

def it_should_do_tfidf_test(self):
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    collection_of_document_term_vectors = []
    vector_index_to_keyword_mapping = []

    parser = None

    def __init__(self, documents = [], transforms = [TFIDF, LSA]):
      self.collection_of_document_term_vectors = []
      self.parser = Parser()
      if len(documents) > 0:
        self._build(documents, transforms)


    def related(self, document_id):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [self._cosine(self.collection_of_document_term_vectors[document_id], document_vector) for document_vector in self.collection_of_document_term_vectors]
        ratings.sort(reverse = True)
        return ratings


    def search(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self._build_query_vector(searchList)

        ratings = [self._cosine(queryVector, documentVector) for documentVector in self.collection_of_document_term_vectors]
        ratings.sort(reverse=True)
        return ratings


    def _build(self, documents, transforms):
      """ Create the vector space for the passed document strings """
      self.vector_index_to_keyword_mapping = self._get_vector_keyword_index(documents)

      matrix = [self._make_vector(document) for document in documents]
      matrix = reduce(lambda matrix,transform: transform(matrix).transform(), transforms, matrix)
      self.collection_of_document_term_vectors = matrix

    def _get_vector_keyword_index(self, document_list):
      """ create the keyword associated to the position of the elements within the document vectors """
      vocabulary_list = self.parser.tokenise_and_remove_stop_words(document_list)
      unique_vocabulary_list = self._remove_duplicates(vocabulary_list)
    
      vector_index={}
      offset=0
      #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
      for word in unique_vocabulary_list:
        vector_index[word] = offset
        offset += 1
      return vector_index  #(keyword:position)


    def _make_vector(self, word_string):
      """ @pre: unique(vectorIndex) """

      vector = [0] * len(self.vector_index_to_keyword_mapping)

      word_list = self.parser.tokenise_and_remove_stop_words(word_string.split(" "))

      for word in word_list:
            vector[self.vector_index_to_keyword_mapping[word]] += 1; #Use simple Term Count Model
      return vector


    def _build_query_vector(self, term_list):
      """ convert query string into a term vector """
      query = self._make_vector(" ".join(term_list))
      return query


    def _remove_duplicates(self, list):
        """ remove duplicates from a list """
        return set((item for item in list))
    
        
    def _cosine(self, vector1, vector2):
      """ related documents j and q are in the concept space by comparing the vectors :
        cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
      return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

