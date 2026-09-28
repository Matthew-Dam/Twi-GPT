
"""
Twi base-model pretraining, with text samples printed every SAMPLE_EVERY epochs.

Needs in GOOGLE_DRIVE_PATH: merged.txt, twi_tokenizer.json (from train_tokenizer.py),
and twi_base_model.py (for save_base_model).
"""
import os, sys, time, math, functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

# ------------------------------ CONFIG ------------------------------
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/"
CORPUS_FILE = "merged.txt"
TOKENIZER_FILE = "twi_tokenizer.json"
MODEL_NAME = "twi_base-v2"
BEST_CKPT = "twi_best.pt"
SAMPLES_LOG = "twi_samples.txt"

MAX_LEN = 128
D_MODEL, N_HEADS, N_LAYERS = 256, 8, 6     # ~7M params with an 8k vocab; drop D_MODEL to 192 if val loss climbs
D_FF = D_MODEL * 4
DROPOUT = 0.1

BATCH_SIZE = 64
NUM_EPOCHS = 30
PEAK_LR = 6e-4
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.1
WARMUP_EPOCHS = 2
PATIENCE = 6            # stop if val loss hasn't improved for this many epochs
VAL_FRACTION = 0.05
SEED = 42

SAMPLE_EVERY = 5        # generate text every N epochs (and at the end)
SAMPLE_TOKENS = 60
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.9
PROMPTS = ["Maakye", "Ghana yɛ", "Me din de", "Ɛnnɛ", "Nyame"]
# --------------------------------------------------------------------


def p(name):
    return os.path.join(GOOGLE_DRIVE_PATH, name)


# ------------------------------- DATA --------------------------------
def load_sequences(path, tokenizer, max_len):
    pad_id = tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.token_to_id("[EOS]")
    bos_id = tokenizer.token_to_id("[BOS]")
    assert pad_id is not None and eos_id is not None, "tokenizer needs [PAD] and [EOS] tokens"

    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f]
    lines = [l for l in lines if len(l) > 10]

    seqs, n_split = [], 0
    for enc in tokenizer.encode_batch(lines):
        ids = list(enc.ids)
        if bos_id is not None and (not ids or ids[0] != bos_id):
            ids = [bos_id] + ids
        if not ids or ids[-1] != eos_id:
            ids.append(eos_id)
        if len(ids) <= max_len:
            seqs.append(ids)
            continue
        # long line: split into chunks instead of throwing the tail away
        n_split += 1
        body, step = ids[:-1], max_len - 1
        for i in range(0, len(body), step):
            chunk = body[i:i + step] + [eos_id]
            if len(chunk) > 8:
                seqs.append(chunk)

    n_tokens = sum(len(s) for s in seqs)
    print(f"{path}: {len(lines):,} lines -> {len(seqs):,} sequences, {n_tokens:,} tokens "
          f"({n_split:,} long lines split into chunks)")
    return seqs


class TwiDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def collate(batch, pad_id):
    """Pad only to the longest sequence in this batch."""
    L = max(len(s) for s in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    pad = ids.eq(pad_id)
    X = ids[:, :-1]
    y = ids[:, 1:].masked_fill(pad[:, 1:], -100)
    return X, y, pad[:, :-1]


# ------------------------------- MODEL -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


def make_attn_mask(x_pad_mask):
    T = x_pad_mask.size(1)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x_pad_mask.device))
    keep = ~x_pad_mask[:, None, None, :]
    return causal[None, None] & keep          # True = may attend


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads, self.d_k, self.d_model, self.dropout = n_heads, d_model // n_heads, d_model, dropout
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q, k, v = self.w_qkv(x).view(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.w_out(out.transpose(1, 2).contiguous().view(B, T, self.d_model))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: more stable to train than post-LN."""
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.attention_norm(x), mask))
        x = x + self.dropout(self.feed_forward(self.feed_forward_norm(x)))
        return x


class TwiTransformer(nn.Module):
    def __init__(self, d_model, n_heads, vocab_size, max_seq_len, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.d_model, self.max_seq_len = d_model, max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.drop = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(d_model, d_ff, n_heads, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight            # tie input/output embeddings

        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.fc:
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.embedding.weight, 0.0, d_model ** -0.5)
        for name, prm in self.named_parameters():         # scale residual projections
            if name.endswith("w_out.weight") or name.endswith("linear_2.weight"):
                nn.init.normal_(prm, 0.0, 0.02 / math.sqrt(2 * num_layers))

    def forward(self, x, pad_mask):
        mask = make_attn_mask(pad_mask)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.drop(self.pos_encoder(x))
        for layer in self.transformer_layers:
            x = layer(x, mask)
        return self.fc(self.final_norm(x))


# ---------------------------- GENERATION -----------------------------
@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens=SAMPLE_TOKENS,
             temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P):
    model.eval()
    eos_id = tokenizer.token_to_id("[EOS]")
    bos_id = tokenizer.token_to_id("[BOS]")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids if prompt else []
    if bos_id is not None:
        ids = [bos_id] + ids
    if not ids:
        return ""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_in = x[:, -model.max_seq_len:]
        logits = model(x_in, torch.zeros_like(x_in, dtype=torch.bool))[0, -1].float()
        logits = logits / max(temperature, 1e-5)
        if top_k and top_k < logits.numel():
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = float("-inf")
        if top_p < 1.0:
            s_logits, s_idx = torch.sort(logits, descending=True)
            probs = F.softmax(s_logits, dim=-1)
            s_logits[probs.cumsum(-1) - probs > top_p] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(0, s_idx, s_logits)
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if nxt.item() == eos_id:
            break
        x = torch.cat([x, nxt.view(1, 1)], dim=1)
    return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)


def show_samples(model, tokenizer, device, epoch, val_loss):
    header = f"\n===== samples after epoch {epoch} (val loss {val_loss:.4f}) ====="
    lines = [header]
    prompts = list(PROMPTS)
    if tokenizer.token_to_id("[BOS]") is not None:
        prompts.append("")                                 # unconditional sample
    for pr in prompts:
        lines.append(f"[{pr or '<BOS>'}] {generate(model, tokenizer, device, pr)}")
    text = "\n".join(lines)
    print(text)
    with open(p(SAMPLES_LOG), "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ------------------------------ TRAINING -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, vocab_size, use_amp, amp_dtype):
    model.eval()
    total, count = 0.0, 0
    for xb, yb, pb in loader:
        xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(xb, pb)
        total += F.cross_entropy(logits.float().view(-1, vocab_size), yb.view(-1),
                                 ignore_index=-100, reduction="sum").item()
        count += (yb != -100).sum().item()
    return total / max(count, 1)


def main():
    torch.manual_seed(SEED)
    if GOOGLE_DRIVE_PATH not in sys.path:
        sys.path.insert(0, GOOGLE_DRIVE_PATH)

    tokenizer = Tokenizer.from_file(p(TOKENIZER_FILE))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    seqs = load_sequences(p(CORPUS_FILE), tokenizer, MAX_LEN)
    perm = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(SEED)).tolist()
    n_val = max(1, int(VAL_FRACTION * len(seqs)))
    val_seqs = [seqs[i] for i in perm[:n_val]]
    train_seqs = [seqs[i] for i in perm[n_val:]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coll = functools.partial(collate, pad_id=pad_id)
    train_loader = DataLoader(TwiDataset(train_seqs), batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=coll, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TwiDataset(val_seqs), batch_size=BATCH_SIZE * 2, shuffle=False,
                            collate_fn=coll, num_workers=2, pin_memory=device.type == "cuda")
    print(f"train sequences {len(train_seqs):,} | val sequences {len(val_seqs):,} | vocab {vocab_size}")

    model = TwiTransformer(D_MODEL, N_HEADS, vocab_size, MAX_LEN, N_LAYERS, D_FF, DROPOUT).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"parameters: {n_params:,} | device: {device}")

    decay = [q for q in model.parameters() if q.requires_grad and q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.requires_grad and q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=PEAK_LR, betas=(0.9, 0.95), fused=device.type == "cuda")

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    train_losses, val_losses = [], []
    best_val, best_train, bad_epochs = float("inf"), float("inf"), 0
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        run_loss, run_tok = 0.0, 0
        for xb, yb, pb in train_loader:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb, pb)
            loss = F.cross_entropy(logits.float().view(-1, vocab_size), yb.view(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ntok = (yb != -100).sum().item()
            run_loss += loss.item() * ntok
            run_tok += ntok

        train_loss = run_loss / max(run_tok, 1)
        val_loss = evaluate(model, val_loader, device, vocab_size, use_amp, amp_dtype)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val, best_train, bad_epochs = val_loss, train_loss, 0
            torch.save(model.state_dict(), p(BEST_CKPT))
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS}: train {train_loss:.4f} | val {val_loss:.4f} "
              f"(ppl {math.exp(val_loss):.1f}) | LR {sched.get_last_lr()[0]:.6f} | "
              f"{time.time() - t0:.0f}s{' *best' if improved else ''}")

        stopping = bad_epochs >= PATIENCE
        if epoch % SAMPLE_EVERY == 0 or epoch == NUM_EPOCHS or stopping:
            show_samples(model, tokenizer, device, epoch, val_loss)
        if stopping:
            print(f"Early stop: no val improvement for {PATIENCE} epochs.")
            break

    # go back to the best checkpoint before saving
    model.load_state_dict(torch.load(p(BEST_CKPT), map_location=device))
    model.eval()
    training_time = time.time() - start

    config = dict(d_model=D_MODEL, n_heads=N_HEADS, vocab_size=vocab_size, max_seq_len=MAX_LEN,
                  num_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT, pad_id=pad_id)
    try:
        from twi_base_model import save_base_model
        save_base_model(model, tokenizer, config, MODEL_NAME, output_dir=GOOGLE_DRIVE_PATH,
                        info={"stage": "pretrain", "data": CORPUS_FILE,
                              "final_loss": round(best_train, 4), "final_val_loss": round(best_val, 4)})
        print(f"Model saved as '{MODEL_NAME}'")
    except Exception as e:
        print(f"save_base_model failed ({e}); best weights are still in {p(BEST_CKPT)}")

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Parameters: {n_params:,} | vocab {vocab_size} | d_model {D_MODEL} | "
          f"heads {N_HEADS} | layers {N_LAYERS}")
    print(f"Epochs run: {len(train_losses)} | time {training_time / 60:.1f} min")
    print(f"Best val loss {best_val:.4f} (ppl {math.exp(best_val):.1f}) | "
          f"train loss at that epoch {best_train:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()