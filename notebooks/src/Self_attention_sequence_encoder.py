import torchtext
from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
from src.BERT_style_modules import BERTStyleEncoder, BERTStyleDecoder
import math
import numpy as np
import tqdm
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 

class SelfAttentionEmbedder(nn.Module):

    def __init__(self, vocab_size=1000, embed_dim=768, att_heads=8, layers=4, dim_feedforward=1024, dropout=0.1, loss_type="cosine"):
        super().__init__()
        self.embedding_size = embed_dim
        self.query_encoder_model = BERTStyleEncoder(vocab_size=vocab_size, dim_model=embed_dim, nhead=att_heads, \
                 num_encoder_layers=layers, d_feed=dim_feedforward, dropout=dropout)
        
        self.query_embedder = self.query_encoder_model.embedder
        self.query_encoder = self.query_encoder_model.encoder
        
        self.doc_encoder_model = BERTStyleDecoder(vocab_size=vocab_size, dim_model=embed_dim, nhead=att_heads, \
                 num_encoder_layers=layers, d_feed=dim_feedforward, dropout=dropout)
        
        self.doc_embedder = self.doc_encoder_model.embedder # nn.Embedding(vocab_size, embed_dim)
        self.doc_decoder = self.doc_encoder_model.decoder
        
        self.loss_type = loss_type
        
        self.stats = {}
        self.init_weights()
    
    def init_train_params(self, lr=0.005, gamma=0.99):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=gamma)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def init_weights(self):
        initrange = 0.1
        
    @property
    def device(self):
        """
        To get a ref to an object run: dict(model.named_parameters()).keys()[0]
        """
        return self.query_encoder_model.embedder.word_embeddings.weight.device
    
    def doc_embedding(self, doc):
        doc_emb = self.doc_embedder(doc) * math.sqrt(self.embedding_size)
        doc_encoding = self.doc_encoder(doc_emb)
        doc_pooled = doc_encoding[0]
        return doc_pooled
    
    def query_embedding(self, query):
        query_emb = self.query_embedder(query) * math.sqrt(self.embedding_size)
        query_encoding = self.query_encoder(query_emb)
        query_pooled = query_encoding[0]
        return query_pooled

    def forward(self, query, doc):        
        query_emb = self.query_embedder(query) * math.sqrt(self.embedding_size)
        doc_emb = self.doc_embedder(doc) * math.sqrt(self.embedding_size)
        
        query_encoding = self.query_encoder(query_emb)
        doc_encoding = self.query_encoder(doc_emb)
        
        query_pooled = query_encoding[0]
        doc_pooled = doc_encoding[0]
        
        return query_pooled, doc_pooled
    
    def similarity(self, queries, docs):
        """
        queries: 2D array of ids dim: Seq_len x Batch
        
        """
    
    def cosine_matrix(self, query_representations, code_representations):
        query_norm = torch.norm(query_representations, dim=-1, keepdim=True) + 1e-10
        code_norm = torch.norm(code_representations, dim=-1, keepdim=True) + 1e-10

        cosine_similarities = torch.mm(torch.div(query_representations,query_norm),
                                        torch.div(code_representations,code_norm).T)
        return cosine_similarities
    
    def softmax_matrix(self, query_representations, code_representations):
        logits = torch.mm(query_representations, code_representations.T)
        return logits
    
    def loss(self, query_representations, code_representations):
        margin = 1.
        
        if self.loss_type == "cosine":
            cosine_similarities = self.cosine_matrix(query_representations, code_representations)
            neg_matrix = torch.diag(torch.full((cosine_similarities.shape[0],), float('-inf'))).to(self.device)

            good_sample_loss = torch.diagonal(cosine_similarities)
            bad_sample_loss = torch.max(cosine_similarities + neg_matrix, dim=-1)[0]
            per_sample_loss = torch.clamp(margin - good_sample_loss + bad_sample_loss, min=0.0)
        
        elif self.loss_type == "softmax":
            logits = self.softmax_matrix(query_representations, code_representations)
            labels = torch.arange(0,code_representations.shape[0]).to(self.device)
            per_sample_loss = self.cross_entropy(logits, labels)
        return torch.mean(per_sample_loss)
    
    def train_step(self, batch):
        query_ids = batch.query
        doc_ids = batch.doc
        
        self.optimizer.zero_grad()
        query_vector, doc_vector = self(query_ids, doc_ids)
        loss = self.loss(query_vector, doc_vector)
        loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        return loss
    
    def data2dataset(self, data, vocab):
        TEXT_FIELD = Field(sequential=True, use_vocab=False, pad_token=vocab.PAD)

        examples = []

        for (query, doc) in tqdm.tqdm(data):
            query_ids = [vocab.CLS] + vocab.encode_input(query)[0] + [vocab.EOS]
            doc_ids = [vocab.CLS] + vocab.encode_input(doc)[0] + [vocab.EOS]

            example = torchtext.data.Example.fromdict({"query":query_ids, 
                                                   "doc":doc_ids},
                                                        fields={"query":("query",TEXT_FIELD), 
                                                                "doc":("doc",TEXT_FIELD)})
            examples.append(example)
        fields = {"query":TEXT_FIELD, "doc":TEXT_FIELD}
        dataset = torchtext.data.Dataset(examples,fields=fields)
        return dataset