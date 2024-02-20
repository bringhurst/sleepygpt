import torch
import torch.nn as nn

class SleepyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super(SleepyGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.TransformerEncoderLayer(embed_dim, nhead=8)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_blocks(x)
        x = self.output_layer(x)
        return x
