#!/usr/bin/env python3
"""
Text Encoding Utility for Time-MMD
====================================
Pre-encodes text descriptions into dense vectors using BERT (or GPT-2).
Run once per domain to create text_embeddings.npy.

Usage:
  python encode_text.py --domain_dir ./data/timemmd/health --model bert
  python encode_text.py --domain_dir ./data/timemmd/energy --model gpt2
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_with_bert(texts: list[str], model_name: str = "bert-base-uncased",
                     batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    """Encode texts using BERT [CLS] token."""
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :]  # [CLS]

        all_embeds.append(cls_embeds.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.concatenate(all_embeds, axis=0)


def encode_with_gpt2(texts: list[str], model_name: str = "gpt2",
                     batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    """Encode texts using GPT-2 last-token pooling."""
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling over non-padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

        all_embeds.append(pooled.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.concatenate(all_embeds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="bert",
                        choices=["bert", "gpt2"])
    parser.add_argument("--model_name", type=str, default=None,
                        help="HuggingFace model name override")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--text_file", type=str, default="text.csv",
                        help="Name of text CSV file in domain_dir")
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir)

    # Load text
    text_path = domain_dir / args.text_file
    if not text_path.exists():
        # Try finding any text file
        for alt in ["text.csv", "texts.csv", "descriptions.csv"]:
            if (domain_dir / alt).exists():
                text_path = domain_dir / alt
                break
        else:
            raise FileNotFoundError(f"No text file found in {domain_dir}")

    logger.info(f"Loading text from {text_path}")
    df = pd.read_csv(text_path)

    # Find text column
    text_col = None
    for col in df.columns:
        if col.lower() in ("text", "description", "content", "narrative"):
            text_col = col
            break
    if text_col is None:
        # Use last column
        text_col = df.columns[-1]

    texts = df[text_col].fillna("").astype(str).tolist()
    logger.info(f"Loaded {len(texts)} text entries from column '{text_col}'")

    # Encode
    if args.model == "bert":
        model_name = args.model_name or "bert-base-uncased"
        logger.info(f"Encoding with BERT ({model_name})...")
        embeddings = encode_with_bert(
            texts, model_name, args.batch_size, args.max_length,
        )
    else:
        model_name = args.model_name or "gpt2"
        logger.info(f"Encoding with GPT-2 ({model_name})...")
        embeddings = encode_with_gpt2(
            texts, model_name, args.batch_size, args.max_length,
        )

    # Save
    out_path = domain_dir / "text_embeddings.npy"
    np.save(str(out_path), embeddings.astype(np.float32))
    logger.info(f"Saved embeddings: shape={embeddings.shape} → {out_path}")


if __name__ == "__main__":
    main()
