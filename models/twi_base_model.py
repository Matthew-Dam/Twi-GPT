# model.py
import torch
import json
import os
from datetime import datetime


def save_base_model(model, tokenizer, config, model_name, info=None):
    """
    Save model weights, config, and tokenizer for later fine-tuning.

    Args:
        model: PyTorch model
        tokenizer: HuggingFace tokenizer
        config: Dict with model hyperparameters
        model_name: String name for the model (folder)
        info: Dict with metadata (stage, data, loss, etc.)
    """

    # Create model directory
    model_dir = f"./{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    # 1. Save model weights
    torch.save(model.state_dict(), f"{model_dir}/model_weights.pth")
    print(f"✓ Saved model weights to {model_dir}/model_weights.pth")

    # 2. Save config
    config_with_metadata = {
        **config,
        "saved_at": datetime.now().isoformat(),
        "info": info or {}
    }

    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config_with_metadata, f, indent=2)
    print(f"✓ Saved config to {model_dir}/config.json")

    # 3. Save tokenizer
    tokenizer.save(f"{model_dir}/tokenizer.json")
    print(f"✓ Saved tokenizer to {model_dir}/tokenizer.json")

    # 4. Save README
    readme = f"""# {model_name}

## Model Info
- **Stage**: {info.get('stage', 'unknown') if info else 'unknown'}
- **Data**: {info.get('data', 'unknown') if info else 'unknown'}
- **Final Loss**: {info.get('final_loss', 'unknown') if info else 'unknown'}
- **Training Time**: {info.get('training_time', 'unknown') if info else 'unknown'}s

## Architecture
- d_model: {config['d_model']}
- n_heads: {config['n_heads']}
- num_layers: {config['num_layers']}
- vocab_size: {config['vocab_size']}
- d_ff: {config['d_ff']}

## Usage

```python
import torch
from model import TwiTransformer, load_base_model

# Load the model
model, tokenizer, config = load_base_model("{model_name}")

# Use for inference
text = "Mo nyinaa"
tokens = tokenizer.encode(text).ids
output = model(torch.tensor([tokens]))
```
"""

    with open(f"{model_dir}/README.md", "w") as f:
        f.write(readme)
    print(f"✓ Saved README to {model_dir}/README.md")

    print(f"\n✅ Model '{model_name}' saved successfully!")
    return model_dir


def load_base_model(model_name):
    """
    Load a saved base model, tokenizer, and config.

    Args:
        model_name: String name of the model folder

    Returns:
        (model, tokenizer, config)
    """
    from tokenizers import Tokenizer

    model_dir = f"./{model_name}"

    # Load config
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)

    # Remove metadata from config (only keep hyperparams)
    info = config.pop("info", {})
    config.pop("saved_at", None)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

    # Load model (you'll need to import TwiTransformer)
    from twi_tran import TwiTransformer  # ← Update path if needed

    model = TwiTransformer(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config.get('dropout', 0.1)
    )

    # Load weights
    model.load_state_dict(torch.load(f"{model_dir}/model_weights.pth",
                                     map_location=torch.device('cpu')))
    model.eval()
    print(f"✓ Loaded model from {model_dir}")

    return model, tokenizer, config