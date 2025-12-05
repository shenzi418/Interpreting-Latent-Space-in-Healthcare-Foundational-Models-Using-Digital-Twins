# Week 4 - Multi-Head Joint Training & Latent Space

## 1. Overview
- zero-shot and baseline result from PTB-XL.
- shared Net1D encoder with two heads (MedalCare 8-class synthetic, PTB-XL 5-superclass real).
- Training script supports single-dataset runs (MedalCare or PTB-XL) and joint MedalCare+PTB-XL via `--joint-datasets medalcare+ptbxl`, with `--multi-head` toggling heads.
- frozen-encoder joint training.
- Added t-SNE latent-space visualization comparing MedalCare vs PTB-XL embeddings from the shared encoder.
- Backward compatibility retained for legacy single-head MedalCare/PTB workflows.

## 2. Results

- PTB-XL baselines (single-dataset, best checkpoints):
  - Linear: test macro AP 0.782, Brier 0.121, ROC-AUC 0.915.
  - MLP: test macro AP 0.791, Brier 0.121, ROC-AUC 0.916.
- Joint MedalCare+PTB-XL (frozen encoder, multi-head):
  - Val MedalCare: AP 0.797, Brier 0.108, ROC-AUC 0.974.
  - Val PTB-XL: AP 0.777, Brier 0.122, ROC-AUC 0.914.
- Zero-shot PTB-XL (foundation checkpoint, no finetuning): macro AP 0.184, Brier 0.560, ROC-AUC 0.318.

## 3. Latent-Space t-SNE (Frozen Encoder Joint Training)

- The visualization script loads a joint multi-head checkpoint, uses the shared encoder to extract latent vectors z for MedalCare test ECGs (synthetic) and PTB-XL test ECGs (real), runs t-SNE to 2D, and plots synthetic (blue) vs real (red), saving under `outputs/`.

- Current interpretation (frozen encoder joint checkpoint): the t-SNE shows two largely disjoint clouds—MedalCare on one side, PTB-XL on the other—with minimal overlap. This suggests the shared latent space remains strongly domain-separated; a domain classifier on z would likely have very high AUROC. Conclusion: frozen-encoder joint training is a baseline but not sufficient to unify synthetic and real manifolds, motivating partial unfreezing (Week 6) and later explicit alignment losses (MMD/CORAL/GRL).

## 4. Next Steps

- Add MMD/CORAL or domain-adversarial loss into the joint loop and repeat alignment analysis.
- Start designing physics / digital-twin probes on the MedalCare latent space and plan how to apply them to PTB-XL.
- (Optional) Add ECE/reliability plots for key labels once best configs are identified.



