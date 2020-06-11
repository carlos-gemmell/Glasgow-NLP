import sys
import lucene
import os
from java.io import File
from java.nio.file import Paths
from lucene import JavaError
import numpy as np
import re

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.document import Document, Field, TextField, FieldType
from org.apache.lucene.search import FuzzyQuery, MultiTermQuery, IndexSearcher
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, FieldInfo, IndexOptions,MultiReader, Term
from org.apache.lucene.store import RAMDirectory, SimpleFSDirectory
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.search.spans import SpanNearQuery, SpanQuery, SpanTermQuery, SpanMultiTermQueryWrapper
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser, QueryParser

from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene import analysis, document, index, queryparser, search, store, util

from src.metrics import nltk_bleu
from src.useful_utils import string_split_v3


class PyLuceneRetriever():
    def __init__(self, index_path=None):
        '''
        This initiates a PyLucene writer class in RAM.
        '''
        try:
            lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        except:
            print("JVM Running")
        
        if index_path:
            self.store = store.SimpleFSDirectory(Paths.get(index_path))
        else:
            self.store = store.RAMDirectory()
        
        self.t2 = FieldType()
        self.t2.setStored(False)
        self.t2.setTokenized(True)
        self.t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        self.analyzer = StandardAnalyzer()
        self.config = IndexWriterConfig(self.analyzer)
        self.config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        self.writer = IndexWriter(self.store, self.config)
        
    def add_doc(self, text):
        '''
        Adds a document to the retrieval engine.
        
        Args:
            text: String: String with words of a single complete document separated by spaces.
        '''
        doc = Document()
        doc.add(Field("field", text, self.t2))        
        self.writer.addDocument(doc)
    
    def add_multiple_docs(self, documents):
        '''
        Adds many documents to the engine efficiently.
        
        Args:
            documents: [String]: a list of strings to be added as individual documents
        '''
        for doc in documents:
            self.add_doc(doc)
        self.writer.commit()
        self.writer.close()
        
    def search(self, query, max_retrieved_docs=3, k1=1.2, b=0.75):
        searcher=IndexSearcher(DirectoryReader.open(self.store))
        searcher.setSimilarity(BM25Similarity(float(k1), float(b)))
        
        queryparser.classic.QueryParser( "fieldname", self.analyzer)
        parser = queryparser.classic.QueryParser("field", self.analyzer)
        
        removelist = "^ "
        query = re.sub(r'[^\w'+removelist+']', '',query)
#         query = QueryParser.escape(query)
        query = parser.parse(query)
        hits = searcher.search(query, max_retrieved_docs).scoreDocs
        doc_ids = [(x.doc, x.score) for x in hits]
        return doc_ids

    def augment_src_with_retrieval(self, data, indexing_data, add_src=False, add_tgt=True, rank_return=0):
        new_data = []
        tgt_indexing_data = [tgt for src,tgt in indexing_data]
        src_indexing_data = [src for src,tgt in indexing_data]

        for src, tgt in data:
            doc_ranking = self.BM25_search(src, k1=4.03, b=0.99)
            new_src = src
            if add_src:
                top_src = src_indexing_data[doc_ranking[rank_return].doc] if len(doc_ranking)>rank_return else ""
                new_src += " RETRIEVAL " + top_src
            if add_tgt:
                top_tgt = tgt_indexing_data[doc_ranking[rank_return].doc] if len(doc_ranking)>rank_return else ""
                new_src += " CODE " + top_tgt
            new_data.append((new_src, tgt))
        return new_data
    
class ReCodeRetriever():
    def __init__(self):
        self.docs = []
    
    def add_doc(self, tok_list):
        '''
        text: [str], list of string tokens
        '''
        assert isinstance(tok_list, list) 
        self.docs.append(tok_list)
    
    def add_multiple_docs(self, documents):
        '''        
        Args:
            documents: [[String]]: a list of list of tokens to be added as individual documents
        '''
        for tok_list in documents:
            self.add_doc(tok_list)
    
    def simi_search(self, query, max_retrieved_docs = 10):
        assert isinstance(query, list) 
        simi_scores = []
        for entry in self.docs:
            simi_scores.append(self.simi(query, entry, True))
        simi_scores = np.array(simi_scores)
        simi_scores -= simi_scores.mean()
        top_indices = np.argsort(simi_scores)[-max_retrieved_docs:][::-1]
        return top_indices
        
    def sentence_distance(self, first_sentence, second_sentence, is_list):
        if not is_list:
            first_sentence = first_sentence.split(' ')
            second_sentence = second_sentence.split(' ')
        m = len(first_sentence)+1
        n = len(second_sentence)+1
        matrix = np.zeros((n, m), dtype=int)

        for i in range(n):
            matrix[i][0] = i  # n rows

        for i in range(m):
            matrix[0][i] = i  # m columns
        # print("-----------")
        for i in range(1, n):
            for j in range(1, m):
                if first_sentence[j-1] == second_sentence[i-1]:
                    penalty = 0
                else:
                    penalty = 1

                # get min
                matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1] + penalty)

        # print matrix[n-1][m-1]
        # print matrix
        return matrix, matrix[n-1][m-1]
    
    
    def simi(self, first_sentence, second_sentence, is_list):
        if not is_list:
            max_score = max(len(first_sentence.split()), len(second_sentence.split()))
            simi = 1.0 - (float(self.sentence_distance(first_sentence,
                                                  second_sentence, False)[1])/float(max_score))
        else:
            max_score = max(len(first_sentence), len(second_sentence))
            simi = 1.0 - (float(self.sentence_distance(first_sentence,
                                                  second_sentence, True)[1])/float(max_score))
        return simi
    

class OracleBLEURetriever():
    def __init__(self, ids_to_keep=20):
        self.ids_to_keep = ids_to_keep
        self.src_store = []
        self.tgt_store = []
        self.matched_queries_index = {} # this matches a src string to the best ids
        
    def add_doc(self, doc):
        '''
        Args:
            text: (String:src, String:tgt)
        '''
        self.src_store.append(doc[0])
        self.tgt_store.append(doc[1])
    
    def add_multiple_docs(self, documents):
        '''
        Adds many documents to the engine efficiently.
        
        Args:
            documents: [(String, String)]: a list of tuples of strings to be added as individual documents
        '''
        for doc in documents:
            self.add_doc(doc)
        
    def search(self, query, tgt, max_retrieved_docs=3):
        
        if query in self.matched_queries_index:
            return self.matched_queries_index[query][:max_retrieved_docs]
        
        BLEU_scores = []
        
        for i in range(len(self.src_store)):
            sample_BLEU = nltk_bleu(string_split_v3(tgt), string_split_v3(self.tgt_store[i]))
            BLEU_scores.append(sample_BLEU)
            
        BLEU_scores = np.array(BLEU_scores)
        arg_BLEU_scores = np.argsort(BLEU_scores)[::-1]
                
        self.matched_queries_index[query] = list(zip(arg_BLEU_scores[:self.ids_to_keep], BLEU_scores[arg_BLEU_scores[:self.ids_to_keep]]))
        return self.matched_queries_index[query]
        