# Building GPT from Scratch

Implementing a GPT language model from scratch, following Andrej Karpathy's tutorial.

## What's implemented
- Character-level tokenizer (65 vocab size)
- Bigram Language Model (baseline)
- Self-Attention Head
- Multi-Head Attention
- FeedForward Layer
- Full Transformer Block
- GPT Language Model

## Results
| Model | Loss |
|-------|------|
| Bigram | 2.37 |
| GPT (3 blocks, 4 heads) | 1.87 |

## Dataset
Shakespeare text (~1.1M characters)

## Tech Stack
Python, PyTorch, Google Colab
