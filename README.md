# Method

## Core Approach
- Weighted $R^2$ score on 53% test set in competition submission: **0.46**.
- Applied **L1 and L2 regularization** to reduce overfitting.

---

# Training Configurations
1. Applied L1 & L2 regularization to reduce overfitting.  
2. Applied CV-Fold & Test Time Augmentation (TODO).  
3. DinoV2 fused with masked tabular (TODO).  

---

# Evaluation Report

## 1. Full Data Training (No Preprocessing)
Full training performed without any preprocessing steps.

### Model Performance (Weighted $R^2$)

| Model                                  | Configuration                   | Weighted $R^2$ |
|----------------------------------------|----------------------------------|-------------------|
| DinoV2-Basic (Image Regression)        | Visual-only regression           | 0.46              |
| ResNet50 (Masked Tabular Fusion)       | Image + masked tabular fusion    | 0.56              |

---

## 2. With Data Preprocessing

### Data Preprocessing Pipeline
1. Removed samples with mismatched **GDM_g** and **Dry_Total_g** values caused by human-entry errors.  
2. Removed outliers detected using the **Interquartile Range (IQR)** method.  

---
# Training command

```python main.py --n-folds 2 --epochs 50 --wandb-api-key API_KEY```

