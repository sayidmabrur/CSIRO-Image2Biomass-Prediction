# Training command
---
```python main.py --n-folds 2 --epochs 50 --wandb-api-key API_KEY```


# TODOS
---
1. grouping K-Fold by tabular existed tabular features
2. Added ensemble method to train dinov3 convnext + dinov3 ViT
3. Added ensembling method Dinov3-convnext-LVD689m with Dinov3-ViT-sat493m


# Problems & Solutions
---
1. High variance problem
- Problem: The current model has high variance, the model has a good prediction in training phase, but in testing environment. It has a bad performance.
- Why: After analyzing the data, the biomass data has unique features for each species. The current spliting method doesn't equally split the data in training & test set. And the dataset probably is too small.
- Solution: Applied stratified K-Fold [TODO]

2. High bias
- Problem: Intuitively, the model probably has high bias, since it performs not very good. It has only 0.4 ~ 0.7 R^2 score on each fold.
- Why: Maybe because dataset is noisy. I'm too lazy to perform preprocessing to select only high quality data or looking for more biomass dataset in the internet.
- Solution: 1. Looking for more dataset & perform heavy preprocessing
            2. Increase the model complexity, added ensembling method