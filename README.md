# Configs
Tunable Hyperparameters:
- initial learning rate: 1e-2 ~ 1e-4
- batch_size: 8/16/32

- 
# Training command
---
```python main.py --n-folds 2 --epochs 50 --wandb-api-key API_KEY```


# TODOS
---
1. grouping K-Fold by tabular existed tabular features
2. Added ensemble method Dinov3-convnext-LVD689m with Dinov3-ViT-sat493m
3. Create a summary & comprehensive analysis between activation function Mish vs SiLU on this data
4. 

# Problems & Solutions
---
1. High variance problem
- Problem: From the summary of K-Fold CV, the model has high variance. It's also caused the model has low weighted R^2 score on the test set. 
- Why: After analyzing the data, the biomass data has unique features for each species. The current spliting method doesn't equally split the data in training & test set. Required more training data so the model could learn to generalize at predicting biomass length.
- Solution: 1. Applied stratified K-Fold [TODO]
            2. Applied test-time augmentation [TODO]

2. High bias
- Problem: Intuitively, the model probably has high bias, since it performs not very good. It has only 0.4 ~ 0.7 R^2 score on each fold.
- Why: Maybe because dataset is noisy. I'm too lazy to perform preprocessing to select only high quality data or looking for more biomass dataset in the internet.
- Solution: 1. Looking for more dataset & perform heavy preprocessing [TODO]
            2. Increase the model complexity, added ensembling method [TODO]

3. Low performance on predicting Dry_Dead_g
- Problem: Data imbalance.
- Why: The current data training, has high amount of clover & green biomass
- Solution: .... (i'll mind about this problem later)