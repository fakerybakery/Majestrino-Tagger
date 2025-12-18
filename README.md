# Majestrino Tagger

Audio tagging with CLIP-style audio-text alignment using Whisper and GTE encoders.

## Installation

```bash
pip install git+https://github.com/fakerybakery/majestrino-tagger.git
```

## Quick Start

```python
from majestrino_tagger import MajestrinoTagger

# Load pretrained model and bundled tags
tagger = MajestrinoTagger.from_pretrained()
tagger.load_tags()  # Uses bundled tags

# Tag an audio file
results = tagger.tag("audio.wav", threshold=50.0, top_n_per_category=3)

for r in results:
    print(f"{r['prob']:.1f}% - {r['label']} ({r['category']})")
```

## Custom Tags

```python
# Load from custom JSON file
tagger.load_tags("my_tags.json")

# Or set tags programmatically
tagger.set_tags(
    tags=["happy upbeat energetic music", "sad melancholic slow music"],
    labels=["Happy", "Sad"],
    categories=["Mood", "Mood"]
)
```

## Advanced Usage

```python
# Get audio embedding
audio_embed = tagger.encode_audio("audio.wav")  # Shape: (1, 768)

# Get text embeddings
text_embeds = tagger.encode_texts(["happy music", "sad music"])  # Shape: (2, 768)

# Compute custom similarity
similarity = audio_embed @ text_embeds.T

# Get raw similarities for all tags
similarities = tagger.get_raw_similarities("audio.wav")
```

## License

MIT
