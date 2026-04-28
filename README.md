# mushroom-toxicity-classifier-using-ML







Problem motivation (safety of mushroom consumption), dataset overview, and the guiding research question (which model gives best recall for poisonous class).

Dataset Description: 61,069 rows, 20 features, binary classification. Identifies 3 quantitative + 17 categorical features. Explains why Label Encoding was chosen over One-Hot (17 nominal columns would explode to 170+ binary columns). Correlation heatmap (Figure 2) shows stem-width and cap-diameter are the strongest predictors. Class distribution (55.5% poisonous / 44.5% edible = mild imbalance). 6-panel EDA bar charts showing per-category class splits.

Preprocessing: Three-tier approach: (a) Drop 4 columns with >80% missing — fabricating 90% of a column defeats the purpose. (b) Mode imputation for 5 columns with 4–62% missing — mode is the only valid central tendency for categorical data. (c) StandardScaler — required for KNN (distances), Logistic Regression (gradient descent stability), and MLP (activation functions).

Splitting: Stratified 80/20 split (48,855 train / 12,214 test). Stratified instead of random because mild class imbalance means naive random splits could skew evaluation.

Models: All 5 required models + K-Means unsupervised. Each comes with a "why this model / why it performs this way" explanation. K-Means ARI = 0.004 (near-random) — this proves the classes are not geometrically separable, which is why non-linear models win.

Comparison:Accuracy bar chart, precision-recall comparison, all 5 confusion matrices, ROC curves, AUC bar chart, and a full summary table. Neural Network wins (#1) with 99.96% accuracy and **perfect recall = 1.000** (zero poisonous mushrooms misclassified as edible).
Conclusion: Explains *why* the gap between non-linear (KNN, MLP, DT) and linear (LR, NB) models exists, challenges faced, and a concrete deployment recommendation.
