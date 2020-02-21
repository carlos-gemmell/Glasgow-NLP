class CustomEmbedder(nn.Module):

    def __init__(self, d_model, vocab_size, dropout=0.1, max_len=512):
        super(CustomEmbedder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.word_embeddings = nn.Embedding(vocab_size,d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        
        self.embedding_layer_norm = LayerNorm(d_model, eps=1e-12)
        
    def forward(self, input_ids):
        
        input_shape = input_ids.shape
        seq_length = input_shape[0]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(1).expand(input_shape)

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings