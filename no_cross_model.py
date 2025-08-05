import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, use_tokenizer=False):
        super(ModalityAdapter, self).__init__()
        if use_tokenizer:
            self.tokenizer = nn.Linear(input_dim, output_dim)
        else:
            self.use_tokenizer = False

    def forward(self, x):
        if self.use_tokenizer:
            return self.tokenizer(x)
        return x

class FusionTransformer(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, fusion_type="concat", pool_type="cls",
                 add_bias_kv=False, dropout=0.1):
        super(FusionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.fusion_type = fusion_type
        self.pool_type = pool_type

        # Define the transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=4*embed_dim,
                                       dropout=dropout, batch_first=True)
            for _ in range(n_layers)]
        )

    def forward(self, latent_tokens, key_padding_mask=None):
        # Apply transformer layers
        for layer in self.transformer_layers:
            latent_tokens = layer(latent_tokens, src_key_padding_mask=key_padding_mask)
        
        # Pooling
        if self.pool_type == "cls":
            return latent_tokens[:, 0, :]
        elif self.pool_type == "mean":
            return latent_tokens.mean(dim=1)

class ModifiedTransformer_Based_Model(nn.Module):
    def __init__(self, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, dropout):
        super(ModifiedTransformer_Based_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        # Encoders for different modalities
        self.text_encoder = nn.TransformerEncoder(d_model=hidden_dim, nhead=n_head, num_layers=1, dropout=dropout)
        self.audio_encoder = nn.TransformerEncoder(d_model=hidden_dim, nhead=n_head, num_layers=1, dropout=dropout)
        self.visual_encoder = nn.TransformerEncoder(d_model=hidden_dim, nhead=n_head, num_layers=1, dropout=dropout)
        
        # Adapters for each modality
        self.text_adapter = ModalityAdapter(D_text, hidden_dim, use_tokenizer=True)
        self.audio_adapter = ModalityAdapter(D_audio, hidden_dim, use_tokenizer=False)
        self.visual_adapter = ModalityAdapter(D_visual, hidden_dim, use_tokenizer=True)
        
        # Fusion Transformer
        self.fusion_transformer = FusionTransformer(embed_dim=hidden_dim, n_heads=n_head, n_layers=1,
                                                     fusion_type="concat", pool_type="cls",
                                                     dropout=dropout)
        
        # Emotion Classifier
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, textf, visuf, acouf):
        # Tokenize and encode modalities
        text_tokens = self.text_adapter(textf)
        audio_tokens = self.audio_adapter(acouf)
        visual_tokens = self.visual_adapter(visuf)

        # Encode using separate encoders
        text_encodings = self.text_encoder(text_tokens.permute(1, 0, 2)).permute(1, 0, 2)
        audio_encodings = self.audio_encoder(audio_tokens.permute(1, 0, 2)).permute(1, 0, 2)
        visual_encodings = self.visual_encoder(visual_tokens.permute(1, 0, 2)).permute(1, 0, 2)

        # Concatenate encodings
        combined_encodings = torch.cat([text_encodings, audio_encodings, visual_encodings], dim=1)

        # Fusion Transformer
        fused_features = self.fusion_transformer(combined_encodings)
        
        # Emotion Classification
        final_out = self.output_layer(fused_features)
        log_prob = F.log_softmax(final_out, 1)
        prob = F.softmax(final_out, 1)
        
        return log_prob, prob

# Example usage
model = ModifiedTransformer_Based_Model(D_text=128, D_visual=64, D_audio=32, n_head=4,
                                       n_classes=5, hidden_dim=256, dropout=0.1)
