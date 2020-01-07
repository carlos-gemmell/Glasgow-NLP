import sys
import lucene
import os
from java.io import File
from java.nio.file import Paths
from lucene import JavaError

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


class PyLuceneRetriever():
    def __init__(self):
        '''
        This initiates a PyLucene writer class in RAM.
        '''
        try:
            lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        except:
            print("JVM Running")
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
        
    def BM25_search(self, query):
        searcher=IndexSearcher(DirectoryReader.open(self.store))
        searcher.setSimilarity(BM25Similarity())
        
        queryparser.classic.QueryParser( "fieldname", self.analyzer)
        parser = queryparser.classic.QueryParser("field", self.analyzer)
        query = QueryParser.escape(query)
        query = parser.parse(query)
        hits = searcher.search(query, 3).scoreDocs
        
        return list(hits)