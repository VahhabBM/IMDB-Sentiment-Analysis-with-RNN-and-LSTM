# model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5, padding_idx=0, cell_type='LSTM'):
        super().__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.cell_type = cell_type.upper()
        
        # Selecting the Recursion Layer Based on the Input Parameter
        if self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, 
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.cell_type == 'RNN':
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("cell_type must be either 'RNN' or 'LSTM'")
            
        # Dropout and classifier with 2 outputs (negative and positive)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2) 
        
    # Forward Layer
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        # Packing padded sequences based on their actual length
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.cell_type == 'LSTM':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)
            
        # Get the last true hidden state
        final_hidden = hidden[-1]
        
        # Applying dropouts and taking the exit list of classes
        final_hidden = self.dropout(final_hidden)
        return self.fc(final_hidden)