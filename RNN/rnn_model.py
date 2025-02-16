import torch
import torch.nn as nn

class StackedBiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5, bidirectional=True, pretrained_embeddings=None):
        super(StackedBiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Αν θέλεις να παγώσεις τα embeddings
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout,
                           bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, output_dim)
        
    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)         # (batch_size, seq_length, embed_dim)
        rnn_out, _ = self.rnn(embedded)        # (batch_size, seq_length, hidden_dim*2)
        # Global max pooling πάνω στον χρόνο (time dimension)
        pooled, _ = torch.max(rnn_out, dim=1)  # (batch_size, hidden_dim*2)
        dropped = self.dropout(pooled)
        output = self.fc(dropped)              # (batch_size, output_dim)
        return output
