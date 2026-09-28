# TwiTransformer

A compact, decoder-only Transformer language model trained from scratch on Twi (Akan), built as the base model for a Twi NLP project. The repository covers the full pipeline: corpus collection and cleaning, tokenizer training, pretraining with periodic text sampling, and checkpointing for second-stage fine-tuning.

> **Status:** This is just a small base model. I was just making an experiment to see how modern Artifitial Intelligence can work with my local language (Akan Twi).

---

## Highlights

- **Trained from scratch** on Twi text, with a tokenizer built for Twi orthography (ɛ, ɔ, Ɛ, Ɔ preserved and NFC-normalised).
- **Modern small-model recipe:** pre-LayerNorm blocks, tied input/output embeddings, fused attention, AdamW with cosine decay, mixed-precision training.
- **Training you can monitor:** the model generates sample text every 5 epochs, so you see what it has learned, not just the loss.
- **Reproducible and resumable-friendly:** fixed seed, best-checkpoint saving, early stopping.

## Model

| | |
|---|---|
| Architecture | Decoder-only Transformer, pre-LayerNorm |
| Parameters | 6,780,928 (≈ 6.8M) |
| Layers / hidden size | 6 / 256 |
| Attention heads | 8 (head dim 32) |
| Feed-forward | 256 → 1024 → 256, GELU |
| Positional encoding | Sinusoidal, absolute |
| Context length | 128 tokens |
| Output head | Tied with token embedding |
| Regularisation | Dropout 0.1, weight decay 0.1 |


# Transformer

<p align="center">
  <img src="./models/Screenshot from 2026-09-28 13-21-20.png" width="700">
</p>

A decoder-only Transformer for Akan Twi.

Attention is causal and also masks padding tokens. Long training lines are split into chunks rather than truncated.

## Tokenizer

Byte-pair encoding trained on the full corpus with a Metaspace pre-tokenizer and decoder.

| | |
|---|---|
| Vocabulary | 8,000 |
| Normalisation | NFC |
| Special tokens | `[PAD]` (id 0), `[UNK]`, `[BOS]`, `[EOS]` |
| Post-processing | `[BOS] $A [EOS]` |

## Data

The training corpus (`merged.txt`) combines:

- Twi Wikipedia (from the official Wikimedia dump), extracted and cleaned
- Twi Bible text
- Additional cleaned Twi text

All sources are sentence-split, deduplicated, and shuffled at sentence level into a single file.

| | |
|---|---|
| Lines | 124,694 |
| Training sequences | 125,014 (after chunking 374 long lines) |
| Tokens | 3,669,636 |
| Train / validation split | 95% / 5%, random, seed 42 |

**Licensing note.** Wikipedia text is CC BY-SA. The Bible was just a downloadable .pdf that i converted to .txt before preprocessing.

## Results

Model was able to maintain tiny range between training loss and validation loss. Loss is per-token cross-entropy over the 8,000-token vocabulary, so it is **not comparable** to models with a different tokenizer.

| Epoch | Train loss | Val loss | Val perplexity |
|---:|---:|---:|---:|
| 1 | 6.7770 | 5.4934 | 243.1 |
| 5 | 4.2713 | 4.1683 | 64.6 |
| 9 | 4.0224 | 4.0004 | 54.6 |
| Final (30) | 3.5313 | 3.7295 | 41.7 |

Validation loss sits slightly below training loss early in training because training loss is measured with dropout enabled and averaged across the epoch while weights are still improving.

### Sample generations

Unedited output from the epoch-5 checkpoint (temperature 0.8, top-k 40, top-p 0.9), shown to illustrate current behaviour. Not yet reviewed by a native speaker.

| Prompt | Output |
|---|---|
| `Ghana yɛ` | Ghana yɛ nea na ɛwɔ hɔ. |
| `Ɛnnɛ` | Ɛnnɛ yi ara ne nnipa a wɔwɔ Ghana. |

## Quick start

```bash
git clone <https://github.com/Matthew-Dam/Twi-GPT.git> && cd <Twi-GPT>
pip install torch tokenizers
```

**1. Prepare the corpus.** Place a one-sentence-per-line `merged.txt` in the working directory (or set `GOOGLE_DRIVE_PATH` in the scripts).

**2. Train the tokenizer**

```bash
python train_tokenizer.py
```

**3. Pretrain**

```bash
python train_twi.py
```

Key settings live in the config block at the top of `train_twi.py`: model size, batch size, epochs, learning rate, `SAMPLE_EVERY`, and sampling parameters. The best checkpoint is saved to `twi_best.pt`; text samples are appended to `twi_samples.txt`.

**4. Generate text**

```python
import torch
from tokenizers import Tokenizer
from train_twi import TwiTransformer, generate

tok = Tokenizer.from_file("twi_tokenizer.json")
model = TwiTransformer(256, 8, tok.get_vocab_size(), 128, 6, 1024, 0.1)
model.load_state_dict(torch.load("twi_best.pt", map_location="cpu"))

print(generate(model, tok, torch.device("cpu"), "Maakye", temperature=0.6))
```

Lower temperatures (0.5–0.7) give more conservative text; higher values give more variety and more errors.

## Training details

| | |
|---|---|
| Optimiser | AdamW, betas (0.9, 0.95), weight decay 0.1 on matrices |
| Learning rate | 6e-4 peak, 2-epoch warm-up, cosine decay to 10% |
| Batch size / epochs | 64 / 30 |
| Precision | bf16 where supported, otherwise fp16 with loss scaling |
| Regularisation | Gradient clipping at 1.0, dropout 0.1 |
| Model selection | Best validation loss; early stopping after 6 epochs without improvement |
| Padding | Dynamic, per batch |

## Repository layout

```
.
├── train_tokenizer.py     # BPE tokenizer training
├── train_twi.py           # model, training loop, sampling
├── twi_base_model.py      # save/load helpers for the base model
├── twi_tokenizer.json     # trained tokenizer
└── twi_samples.txt        # generated samples per checkpoint
```

## Limitations

- **Small and early.** About 3.7M tokens is little data. The model produces fluent-looking short sentences but is not reliable for meaning or facts.
- **Domain bias.** The corpus is mostly encyclopedic and religious text, so outputs skew toward wiki and Bible style and the model has seen little conversational Twi.
- **No code-switching yet.** The corpus contains almost no Twi–English mixed text.
- **Sentence-level context.** Training data is shuffled by sentence, so the model does not learn discourse beyond a single sentence.
- **Noisy source text.** Some wiki artefacts and sentence-splitting errors remain; a filtering pass is planned.
- **Validation is optimistic.** Near-duplicate lines can appear in both splits, so held-out loss may understate true error.
- **Not evaluated by native speakers.** Treat all output as unverified, and do not use it for anything requiring accuracy.
- **Dialect and orthography.** Sources mix varieties of Twi, and spelling conventions are not fully normalised.

## Roadmap

- [ ] Filter noisy lines and rebuild the corpus
- [ ] Add web-crawled and conversational Twi text
- [ ] Add code-switched Twi–English data
- [ ] Document-level validation split
- [ ] Native-speaker evaluation of samples
- [ ] Second-stage fine-tuning from the saved base model
- [ ] Resumable training checkpoints

## Acknowledgements

Twi Wikipedia contributors and the Wikimedia Foundation; the PyTorch and Hugging Face `tokenizers` projects.

## License

MIT License

Copyright (c) 2026 Matthew-Dam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

`<Matthew Damptey>` · `<+233592612332, dampteymatthew13579@gmail.com>`
