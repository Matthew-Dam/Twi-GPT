
"""
Train a BPE tokenizer on the full merged Twi corpus.

    pip install tokenizers
    python train_tokenizer.py
"""
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers

GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/"
CORPUS = os.path.join(GOOGLE_DRIVE_PATH, "merged.txt")
OUT = os.path.join(GOOGLE_DRIVE_PATH, "twi_tokenizer.json")
VOCAB_SIZE = 8000          # 4000-8000 suits a corpus of this size

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.normalizer = normalizers.NFC()                 # one canonical form for ɛ / ɔ
tok.pre_tokenizer = pre_tokenizers.Metaspace()
tok.decoder = decoders.Metaspace()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],   # [PAD] gets id 0
    show_progress=True,
)
tok.train([CORPUS], trainer)

tok.post_processor = processors.TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[("[BOS]", tok.token_to_id("[BOS]")), ("[EOS]", tok.token_to_id("[EOS]"))],
)
tok.save(OUT)

# quick sanity check
s = "Maakye. Wo ho te dɛn?"
enc = tok.encode(s)
print("vocab size:", tok.get_vocab_size())
print("tokens:", enc.tokens)
print("decoded:", tok.decode(enc.ids, skip_special_tokens=True))