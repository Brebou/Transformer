import Model_Blocks.Encoder as enc
import Model_Blocks.Decoder as dec
import torch.nn as nn
import torch

def positional_encoding(context_size, embedding_size):
    pe = torch.zeros(context_size, embedding_size)
    position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  
    return pe

class Transformer(nn.Module):
    def __init__(self,
        length_vocab_entry,
        length_vocab_target,
        embedding_size = 128,
        dropout_rate = 0.1,
        head_size = 64,
        num_heads = 4,
        n_encoder_blocks = 4,
        n_decoder_blocks = 4,
        max_context_size = 64
        ):
        super().__init__()
        self.encoder = enc.Encoder(num_heads, embedding_size, head_size, embedding_size//num_heads, dropout_rate, n_encoder_blocks)
        self.decoder = dec.Decoder(num_heads, embedding_size, head_size, dropout_rate, n_decoder_blocks)   
        self.positional_encoding = positional_encoding(max_context_size, embedding_size)
        self.embedding_input = nn.Embedding(length_vocab_entry, embedding_size)
        self.embedding_output = nn.Embedding(length_vocab_target, embedding_size)
        self.output_layer = nn.Linear(embedding_size, length_vocab_target)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
        x,
        y,
        mask_input = None,
        ):


        entry_embeddings = self.embedding_input(x) + self.positional_encoding[:, :x.size(1), :].to(x.device)
        target_embeddings = self.embedding_output(y) + self.positional_encoding[:, :y.size(1), :].to(y.device)

        encoder_output = self.encoder(entry_embeddings, mask=mask_input)
        decoder_output = self.decoder(target_embeddings, encoder_output, cross_attention_mask=mask_input)            
        
        output = self.output_layer(decoder_output)
        output = self.dropout(output)
        output = decoder_output @ self.embedding_output.weight.t()

        return output