# Multimodal Product Price Predictor

**Amazon ML Challenge 2025 · AIR 38 among 9,000+ teams**

Predicts an e-commerce product's price from two inputs: its **catalog text** (title, description, pack size) and its **product image**.

## Approach

```
catalog text ──► BERT (bert-base-uncased) ──► pooled 768-d ──► Linear 512 ─┐
                                                                            ├─► concat 1024 ─► MLP (256, ReLU, Dropout) ─► log(price)
product image ─► ViT (vit-base-patch16-224) ─► pooled 768-d ──► Linear 512 ─┘
```

- **Why ViT for images:** self-attention over 16×16 patches captures global context (packaging, product type, brand cues) instead of only local texture.
- **Why SMAPE as the loss:** it is the challenge's evaluation metric and is scale-independent, so cheap and expensive products count equally.
- **Target:** the network predicts `log(price)`, which handles the heavily skewed price distribution; `exp` maps it back and keeps every prediction positive.
- **Optimisation:** AdamW (lr 2e-5, weight decay 0.01) with cosine-annealing learning rate.

## Files

| File | Purpose |
|---|---|
| `multimodal_price_predictor.ipynb` | Data loading, model, training with validation SMAPE, submission |
| `download_images.py` | Downloads all product images once (parallel), so training doesn't fetch over HTTP |
| `requirements.txt` | Dependencies |

## Run

1. Get `train.csv` and `test.csv` from the Amazon ML Challenge 2025 student resources and put them in `dataset/` (the data is not redistributed here).
2. `pip install -r requirements.txt`
3. `python download_images.py --data dataset --out images`
4. Open the notebook and run all cells. It writes `best_model.pth` and `test_out.csv`.

A GPU is strongly recommended (BERT-base + ViT-base, ~200M parameters).

## Team

Built for the Amazon ML Challenge 2025 by Parth Jadhav and team.

## License

Code: MIT, see [LICENSE](LICENSE). The challenge dataset belongs to Amazon and is not included.
