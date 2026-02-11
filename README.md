# User classification
To enable contextual decision-making in the bandit framework, a supervised classification model was developed to predict the user category (User1, User2, User3) using the features provided in train_users.csv. The predicted user category serves as the context for the contextual bandit recommendation system as required in the assignment . </br></br>

## Data Pre-processing</br>
Features used: all columns except user_id, label, browser_version, region_code.</br>
Target: label (cast to string).</br>
Validation setup: StratifiedKFold with 5 folds (shuffle=True, random_state=42).</br>
## Pipelines:</br>
Tree models: SimpleImputer(strategy="median")</br>
Linear/SVM/MLP models: SimpleImputer + StandardScaler</br>
PCA experiment: SimpleImputer + StandardScaler + PCA(n_components=5)</br>

## Experiments Tried
Models (without PCA): Decision Tree, Random Forest, HistGradientBoosting, Logistic Regression, SVM (RBF), MLP.</br>
Models (with PCA=5): same six models with PCA preprocessing.</br>
Metrics: Accuracy and Macro-F1 via cross-validation.</br>

## Best Accuracy
#### Without PCA (best): HistGradientBoosting </br>
Accuracy: 0.8235 ± 0.0346</br>
Macro-F1: 0.8212 ± 0.0350</br>
#### Without PCA (best): RandomForest </br>
Accuracy: 0.8225 ± 0.029707</br>
Macro-F1: 0.818340 ± 0.029490</br>
#### With PCA (best): SVM_RBF_PCA5</br>
Accuracy: 0.7895 ± 0.0233</br>
Macro-F1: 0.7843 ± 0.0238</br>

#### Model for Test Predictions : RandomForest and test results are saved. 

# Contextual Multi Arm Bandit
For each context (user1, user2, user3), a separate 4-arm bandit was trained with arms: [ENTERTAINMENT, EDUCATION, TECH, CRIME].
## Epsilon-Greedy
At each step:</br>
With probability epsilon, select a random arm (exploration).</br>
With probability 1 - epsilon, select the arm with highest current Q (exploitation).</br>
Sample reward using sampler.sample(j) with context-specific arm index mapping.</br>
Update action value incrementally:</br>
$Q(a) \leftarrow Q(a) + \frac{R - Q(a)}{N(a)}$
</br>
Experiments were run for epsilon ∈ {0.01, 0.05, 0.1} over T = 10,000.</br>
Observed Expected Reward Distribution (Q-values)</br>
#### epsilon=0.01
user1: best = EDUCATION (0.8593) </br>
user2: best = CRIME (8.1263)</br>
user3: best = TECH (5.7128)</br>
#### epsilon=0.05
user1: best = EDUCATION (0.8723)</br>
user2: best = CRIME (8.1286)</br>
user3: best = TECH (5.7083)</br>
#### epsilon=0.1
user1: best = EDUCATION (0.8626)</br>
user2: best = CRIME (8.1273)</br>
user3: best = TECH (5.7114)</br>
#### Inference from Graphs (Avg Reward vs Time)
All three epsilon settings converge to similar long-run performance in each context. </br>
Best arm identity is stable across epsilon values: </br>
user1 → EDUCATION</br>
user2 → CRIME</br>
user3 → TECH</br>
Higher epsilon increases random exploration, which typically slows stable exploitation.</br>
Very low epsilon reduces exploration and can converge quickly but risks insufficient sampling early.</br>
A mid-range epsilon (0.05 in the recommendation pipeline) is a practical exploration-exploitation tradeoff. </br>

#### Epsilon-Greedy: Q-Value Summary Across Contexts and Categories

| Epsilon | User  | Entertainment | Education | Tech   | Crime  |
|---|---|---:|---:|---:|---:|
| 0.01 | user1 | -6.8724 | 0.8593 | -0.2972 | -4.2370 |
| 0.01 | user2 | 6.3414 | 3.8627 | 2.6627 | 8.1263 |
| 0.01 | user3 | -1.9097 | -8.8094 | 5.7128 | -0.4805 |
| 0.05 | user1 | -7.4609 | 0.8723 | -0.6929 | -4.2305 |
| 0.05 | user2 | 6.0075 | 3.9385 | 2.6959 | 8.1286 |
| 0.05 | user3 | -1.9547 | -9.0380 | 5.7083 | -0.4819 |
| 0.10 | user1 | -7.1092 | 0.8626 | -0.6341 | -4.1020 |
| 0.10 | user2 | 6.0714 | 3.9167 | 2.7436 | 8.1273 |
| 0.10 | user3 | -1.8751 | -8.7741 | 5.7114 | -0.4726 |





