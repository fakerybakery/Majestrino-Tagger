"""
Majestrino Tagger Pipeline

High-level pipeline for audio tagging that handles:
- Model loading and weight initialization
- Audio preprocessing
- Tag encoding and caching
- Inference with probability scoring
"""

import json
import math
import warnings
from importlib import resources
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
import torch.nn.functional as F
import torchaudio
from transformers import WhisperFeatureExtractor, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from .model import AudioTextAlignmentModel

# Suppress torchaudio deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


class MajestrinoTagger:
    """
    High-level pipeline for tagging audio files with semantic labels.
    
    Example:
        >>> tagger = MajestrinoTagger.from_pretrained()
        >>> tagger.load_tags("tags.json")
        >>> results = tagger.tag("audio.wav")
        >>> for r in results:
        ...     print(f"{r['prob']:.1f}% - {r['label']}")
    """
    
    DEFAULT_REPO_ID = "ChristophSchuhmann/Majestrino_0.11_alpha"
    DEFAULT_FILENAME = "model.safetensors"
    SAMPLE_RATE = 16000
    TARGET_LENGTH = 480000  # 30 seconds at 16kHz
    
    def __init__(self, model: AudioTextAlignmentModel, device: str = None):
        """
        Initialize the tagger with a model.
        
        Args:
            model: Initialized AudioTextAlignmentModel
            device: Device to run inference on. Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        
        # Feature extractor for audio preprocessing
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            AudioTextAlignmentModel.WHISPER_MODEL
        )
        
        # Tokenizer for text encoding
        self.tokenizer = AutoTokenizer.from_pretrained(
            AudioTextAlignmentModel.TEXT_MODEL
        )
        
        # Tag data (populated by load_tags)
        self._tags: List[str] = []
        self._tag_labels: List[str] = []
        self._tag_categories: List[str] = []
        self._tag_embeddings: Optional[torch.Tensor] = None
    
    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str = None,
        filename: str = None,
        device: str = None
    ) -> "MajestrinoTagger":
        """
        Load a pretrained MajestrinoTagger from HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repo ID. Defaults to Majestrino_0.11_alpha.
            filename: Weights filename in the repo.
            device: Device to run on. Auto-detected if None.
            
        Returns:
            Initialized MajestrinoTagger ready for use.
        """
        repo_id = repo_id or cls.DEFAULT_REPO_ID
        filename = filename or cls.DEFAULT_FILENAME
        
        # Initialize model
        model = AudioTextAlignmentModel()
        
        # Download and load weights
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model.load_state_dict(load_file(model_path), strict=False)
        
        return cls(model, device)
    
    def load_tags(self, tags_path: Union[str, Path] = None) -> "MajestrinoTagger":
        """
        Load tags from a JSON file.
        
        If no path is provided, loads the bundled default tags.
        
        Expected JSON format:
            {
                "Category Name": [
                    {"label": "Display Label", "text": "Description for embedding"},
                    ...
                ],
                ...
            }
        
        Args:
            tags_path: Path to the tags JSON file. If None, uses bundled tags.
            
        Returns:
            Self for method chaining.
        """
        if tags_path is None:
            # Load bundled tags
            try:
                # Python 3.9+
                with resources.files("majestrino_tagger").joinpath("tags.json").open() as f:
                    data = json.load(f)
            except AttributeError:
                # Python 3.8 fallback
                with resources.open_text("majestrino_tagger", "tags.json") as f:
                    data = json.load(f)
        else:
            with open(tags_path, 'r') as f:
                data = json.load(f)
        
        self._tags = []
        self._tag_labels = []
        self._tag_categories = []
        
        for category, items in data.items():
            for item in items:
                self._tags.append(item["text"])
                self._tag_labels.append(item["label"])
                self._tag_categories.append(category)
        
        # Pre-compute tag embeddings
        self._encode_tags()
        
        return self
    
    def set_tags(
        self, 
        tags: List[str], 
        labels: List[str] = None, 
        categories: List[str] = None
    ) -> "MajestrinoTagger":
        """
        Set tags programmatically.
        
        Args:
            tags: List of tag descriptions for embedding.
            labels: Optional display labels. Defaults to tags.
            categories: Optional category names. Defaults to "default".
            
        Returns:
            Self for method chaining.
        """
        self._tags = tags
        self._tag_labels = labels or tags
        self._tag_categories = categories or ["default"] * len(tags)
        
        self._encode_tags()
        
        return self
    
    def _encode_tags(self):
        """Pre-compute embeddings for all tags."""
        inputs = self.tokenizer(
            self._tags, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            self._tag_embeddings = self.model.encode_text(
                inputs.input_ids.to(self.device),
                inputs.attention_mask.to(self.device)
            )
    
    def _preprocess_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess audio file for inference.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Mel spectrogram features tensor.
        """
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sr != self.SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(wav)
        
        wav = wav.squeeze()
        
        # Pad or trim to target length (30 seconds)
        if wav.shape[0] < self.TARGET_LENGTH:
            wav = F.pad(wav, (0, self.TARGET_LENGTH - wav.shape[0]))
        else:
            wav = wav[:self.TARGET_LENGTH]
        
        # Convert to mel spectrogram
        inputs = self.feature_extractor(
            wav.numpy(),
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        )
        
        return inputs.input_features
    
    @staticmethod
    def _softmax_with_temperature(scores: List[float], temperature: float = 100) -> List[float]:
        """Apply softmax with temperature scaling."""
        exps = [math.exp(s * temperature) for s in scores]
        sum_exp = sum(exps)
        return [e / sum_exp * 100 for e in exps]
    
    def tag(
        self,
        audio_path: Union[str, Path],
        threshold: float = 50.0,
        top_n_per_category: int = 3,
        temperature: float = 100.0
    ) -> List[Dict]:
        """
        Tag an audio file.
        
        Args:
            audio_path: Path to the audio file.
            threshold: Minimum probability threshold (0-100).
            top_n_per_category: Maximum tags to return per category.
            temperature: Softmax temperature for probability scaling.
            
        Returns:
            List of dicts with keys: label, tag, sim, category, prob
            Sorted by probability descending.
        """
        if self._tag_embeddings is None:
            raise RuntimeError("No tags loaded. Call load_tags() or set_tags() first.")
        
        # Preprocess audio
        features = self._preprocess_audio(audio_path)
        
        # Encode audio
        with torch.no_grad():
            audio_embed = self.model.encode_audio(features.to(self.device))
        
        # Compute similarities
        similarities = (audio_embed @ self._tag_embeddings.T).squeeze()
        
        # Group by category
        category_results: Dict[str, List[Dict]] = {}
        for i, (tag, label, category) in enumerate(
            zip(self._tags, self._tag_labels, self._tag_categories)
        ):
            score = similarities[i].item()
            if category not in category_results:
                category_results[category] = []
            category_results[category].append({
                "label": label,
                "tag": tag,
                "sim": score,
                "category": category
            })
        
        # Apply softmax within each category
        results = []
        for category, items in category_results.items():
            sims = [item["sim"] for item in items]
            probs = self._softmax_with_temperature(sims, temperature)
            
            for item, prob in zip(items, probs):
                item["prob"] = prob
            
            items.sort(key=lambda x: x["prob"], reverse=True)
            
            for item in items[:top_n_per_category]:
                if item["prob"] >= threshold:
                    results.append(item)
        
        results.sort(key=lambda x: x["prob"], reverse=True)
        
        return results
    
    def get_raw_similarities(self, audio_path: Union[str, Path]) -> Dict[str, float]:
        """
        Get raw cosine similarities for all tags.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Dictionary mapping tag labels to similarity scores.
        """
        if self._tag_embeddings is None:
            raise RuntimeError("No tags loaded. Call load_tags() or set_tags() first.")
        
        features = self._preprocess_audio(audio_path)
        
        with torch.no_grad():
            audio_embed = self.model.encode_audio(features.to(self.device))
        
        similarities = (audio_embed @ self._tag_embeddings.T).squeeze()
        
        return {
            label: similarities[i].item()
            for i, label in enumerate(self._tag_labels)
        }
    
    def encode_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Get the audio embedding for a file.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Audio embedding tensor of shape (1, 768).
        """
        features = self._preprocess_audio(audio_path)
        
        with torch.no_grad():
            return self.model.encode_audio(features.to(self.device))
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Get text embeddings for a list of strings.
        
        Args:
            texts: List of text strings to encode.
            
        Returns:
            Text embeddings tensor of shape (len(texts), 768).
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            return self.model.encode_text(
                inputs.input_ids.to(self.device),
                inputs.attention_mask.to(self.device)
            )

