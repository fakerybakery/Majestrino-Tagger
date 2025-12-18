"""
Majestrino Tagger - Audio Tagging with CLIP-style Audio-Text Alignment

A library for semantic audio tagging using a CLIP-style model that aligns
Whisper audio embeddings with GTE text embeddings.

Example:
    >>> from majestrino_tagger import MajestrinoTagger
    >>> 
    >>> # Load pretrained model
    >>> tagger = MajestrinoTagger.from_pretrained()
    >>> 
    >>> # Load tags from JSON
    >>> tagger.load_tags("tags.json")
    >>> 
    >>> # Tag an audio file
    >>> results = tagger.tag("audio.wav", threshold=50.0)
    >>> for r in results:
    ...     print(f"{r['prob']:.1f}% - {r['label']}")

For more control, you can also use the model directly:
    >>> from majestrino_tagger import AudioTextAlignmentModel
    >>> model = AudioTextAlignmentModel()
"""

__version__ = "0.1.0"

from .model import AudioTextAlignmentModel
from .pipeline import MajestrinoTagger

__all__ = [
    "AudioTextAlignmentModel",
    "MajestrinoTagger",
]

