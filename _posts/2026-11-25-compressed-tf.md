---
title: "Compressed cross attention transformer"
date: 2026-11-25
---

Thought I'll share this. I think there are many variations of this already. The intuition is to have a x * 3 and a 3 * x matrix instead of a x^2 matrix. So the memory usage expands linearly.

The caveat is that the query, key, value tokens all have to interact with the module's token layer (or compressed token) first instead of other tokens directly. However, if implemented with recursion, it should learn to update the data in a way where it allows the token to extract the next important feature with every iteration.

Including the token in the inputs themselves (as done here) performs better. Unsure why, but its possibly helping set some context for the model.

In this case, i use `y` as my query, and `x` as the key and value input. But thats's just a quirk, and should be modifiable.

    class CompressedTransformerLayer(nn.Module):
        def __init__(self, embedding_dim=64, heads=8, dropout=0.0, num_compressed_tokens=3):
            super().__init__()
    
            self.num_compressed_tokens = num_compressed_tokens
            self.token_layer = nn.Parameter(torch.randn(num_compressed_tokens, embedding_dim) * 0.02)
            self.compression_transformer = TransformerLayer(embedding_dim=embedding_dim, heads=heads, dropout=dropout)
            self.decompression_transformer = TransformerLayer(embedding_dim=embedding_dim, heads=heads, dropout=dropout)
    
        def forward(self, x):
            B, N, _ = x.shape
    
            #Compressing
            y = self.token_layer.unsqueeze(0).expand(B, -1, -1)
            x_y_combined = torch.cat([x, y], dim=1)
    
            y, _ = self.compression_transformer(
                x=x_y_combined,
                y=y,
            )
    
            #Decompressing
            y, _ = self.decompression_transformer(
                x=y,
                y=x,
            )
    
            return y

    
    #Test
    result = CompressedTransformerLayer()(
        torch.randn(32, 80, 64),
    )
    
    print(result.shape)


