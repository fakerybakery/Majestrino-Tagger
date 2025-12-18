"""
Majestrino Audio-Text Alignment Model

Core model architecture that aligns audio embeddings with text embeddings
using Whisper encoder and GTE text encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, AutoModel


class AudioTextAlignmentModel(nn.Module):
    """
    Audio-Text alignment model that projects Whisper audio embeddings 
    into the same space as GTE text embeddings.
    
    Architecture:
        - Audio: Whisper encoder -> MLP projector -> L2 normalized embedding
        - Text: GTE encoder -> L2 normalized embedding
    """
    
    WHISPER_MODEL = "laion/BUD-E-Whisper"
    TEXT_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
    
    def __init__(self, whisper_model: str = None, text_model: str = None):
        """
        Initialize the audio-text alignment model.
        
        Args:
            whisper_model: HuggingFace model ID for Whisper. Defaults to laion/BUD-E-Whisper.
            text_model: HuggingFace model ID for text encoder. Defaults to Alibaba-NLP/gte-base-en-v1.5.
        """
        super().__init__()
        
        whisper_model = whisper_model or self.WHISPER_MODEL
        text_model = text_model or self.TEXT_MODEL
        
        # Whisper encoder for audio
        self.whisper_full = WhisperModel.from_pretrained(whisper_model)
        self.audio_encoder = self.whisper_full.encoder
        
        # Projects Whisper embeddings (768) to GTE space (768)
        self.projector = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Linear(2048, 768)
        )
        
        # GTE text encoder
        self.text_model = AutoModel.from_pretrained(
            text_model,
            trust_remote_code=True
        )

    def encode_audio(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio mel spectrogram features to embedding.
        
        Args:
            features: Whisper mel spectrogram tensor of shape (batch, 80, 3000)
            
        Returns:
            L2-normalized audio embeddings of shape (batch, 768)
        """
        out = self.audio_encoder(features).last_hidden_state.mean(dim=1)
        return F.normalize(self.projector(out), p=2, dim=1)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode tokenized text to embedding.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            L2-normalized text embeddings of shape (batch, 768)
        """
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.last_hidden_state.mean(dim=1), p=2, dim=1)
    
    def forward(self, audio_features: torch.Tensor = None, 
                input_ids: torch.Tensor = None, 
                attention_mask: torch.Tensor = None):
        """
        Forward pass for computing audio and/or text embeddings.
        
        Args:
            audio_features: Optional mel spectrogram features
            input_ids: Optional token IDs
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with 'audio_embeds' and/or 'text_embeds' keys
        """
        result = {}
        
        if audio_features is not None:
            result['audio_embeds'] = self.encode_audio(audio_features)
            
        if input_ids is not None and attention_mask is not None:
            result['text_embeds'] = self.encode_text(input_ids, attention_mask)
            
        return result

