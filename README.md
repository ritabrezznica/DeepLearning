## PREPROCESSING
- Bipolar {-1, 1} for categoricals, NOT binary {0, 1} — lectures say "bipolar representation helps the network generalise more" because binary zeros kill weight updates
- One-hot encoded EDUCATION and MARRIAGE first, then converted to bipolar — avoids treating them as ordinal (EDUCATION=1 is not "less than" EDUCATION=3)
- Z-score standardisation on continuous features — lectures say "normalise the input features"; without it LIMIT_BAL (up to 1M) dominates AGE (21-79)
- Scaler fit ONLY on training data to prevent data leakage
- Cleaned undocumented values: EDUCATION 0,5,6 merged into 4; MARRIAGE 0 merged into 3
- Feature engineering: added AVG_PAY_DELAY (0.285 correlation), MAX_PAY_DELAY (0.333), NUM_LATE (0.099) — these summarise payment history into informative signals. UTIL_RATIO and PAY_RATIO had near-zero correlation because they were computed on already-scaled values (acknowledge this as a limitation)

## ARCHITECTURE CHOICE
- Multi-Layer Feed-Forward Network with [32, 16] hidden layers, 1,633 parameters
- Lecture rule of thumb: (input + output) / 2 = (33+1)/2 ≈ 17 neurons — our small [16] is closest but medium [32, 16] performed better
- Tested 9 configurations: 3 sizes × 3 activations. Medium beat both small and large
- Larger networks [64, 32] with ~4,400 params did NOT improve F1 — supports the handout's point: "If Group A gets 85% F1 with 1,000 parameters and Group B gets 86% with 1,000,000, Group A is technically superior"
Pyramid shape (32→16) follows lecture recommendation: earlier layers should have more neurons than later ones

## ACTIVATION FUNCTION
- Chose SELU — lecture ranking: SELU > ELU > LeakyReLU > ReLU > tanh > logistic
- SELU confirmed as best in our experiments (0.527 F1) and trained fastest (33 epochs vs 63-131 for others)
- SELU is self-normalising: keeps activations at mean=0, std=1 without needing Batch Normalisation
- Because of SELU: no BatchNorm (would break self-normalisation), AlphaDropout instead of regular Dropout, LeCun weight initialisation

## TRAINING STRATEGY
- Loss: BCEWithLogitsLoss with pos_weight=3.52 — dataset is imbalanced (78% no-default, 22% default), pos_weight up-weights the minority class so the model doesn't just predict "no default" for everything
- Optimizer: Adam — lectures describe it as combining momentum and RMSProp. Tested AdamW (weight decay) but it hurt performance because SELU already regularises implicitly
- LR Schedule: ReduceLROnPlateau — halves the learning rate when validation loss stops improving. Tested 1cycle (lecture-recommended default) but it produced noisy, unstable training curves. Plateau gave smoother convergence and better generalisation
- Early stopping: patience=30 on validation loss — prevents overfitting, lectures say "training stops when the error for the validation set drops to a minimum"
- Batch size: 32 — lectures say small batches (2-32) lead to better generalisation

## GENERALISATION PROOF
- Show the plateau Training vs Validation Loss plot
- Both curves converge together with a small gap — no significant overfitting
- F1 shows clear upward trend from 0.515 to 0.530
- Model converged at epoch 33 — efficient training
Compare with 1cycle plot if asked: 1cycle was noisier and less stable despite slightly higher peak F1

## RESULTS
- Validation F1: 0.527 (during experiments)
- Test F1: 0.517 (held-out set, never seen during training)
- After retraining on full data (24,000 samples instead of 19,200): Test F1 improved to 0.527
- 1,633 parameters — very efficient model
- Recall on defaulters: 61%, Precision: 45%
