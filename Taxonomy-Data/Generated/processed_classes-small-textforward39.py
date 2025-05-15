import torch
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100,
                 embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5,
                 embedding_matrix=None, freeze_embedding_layer=False):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = 1
        self.num_kernels = len(kernel_heights)
        self.pool_sizes = [(max_seq_length - k, 1) for k in kernel_heights]
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

        self.embedding = nn.Embedding(
            vocabulary_size, embed_dim, padding_idx=padding_idx
        )

        if embedding_matrix is not None:
            self.embedding = self.embedding.from_pretrained(embedding_matrix.float(),
                                                            padding_idx=padding_idx)

        self.embedding.weight.requires_grad = not freeze_embedding_layer

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=(k, embed_dim)
                )
                for k in kernel_heights
            ]
        )
        self.pools = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=pool_size)
                for pool_size in self.pool_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.out_channels * self.num_kernels, self.num_classes)

    def forward(self, x):
        # Embedding the input sequences
        x = self.embedding(x)  # Shape: (batch_size, max_seq_length, embed_dim)
        
        # Add a channel dimension for convolution
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, max_seq_length, embed_dim)
        
        # Apply convolutional layers followed by max-pooling
        conv_results = []
        for conv, pool in zip(self.convs, self.pools):
            conv_out = F.relu(conv(x))  # Shape: (batch_size, out_channels, H_out, 1)
            pooled_out = pool(conv_out).squeeze(3)  # Shape: (batch_size, out_channels, 1)
            conv_results.append(pooled_out.squeeze(2))  # Shape: (batch_size, out_channels)
        
        # Concatenate pooled features from different kernels
        x = torch.cat(conv_results, dim=1)  # Shape: (batch_size, out_channels * num_kernels)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through the fully connected layer
        logits = self.fc(x)  # Shape: (batch_size, num_classes)
        
        return logits