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

![Epsolon-Greedy plots](https://github.com/Varchita-Beena/lab3-contextual-bandit/blob/Varchita_D2025PHIL0004/e_greedy.png)

## UCB

$$
\text{UCB}_a = Q_a + C \sqrt{\frac{\ln(t)}{N_a}}
$$

- `Q_a`: current estimated reward of arm `a` (exploitation term)
- `N_a`: number of times arm `a` was chosen
- `ln(t)/N_a` term: uncertainty bonus (exploration term)
- `C`: exploration strength hyperparameter

Arms with high reward stay preferred (`Q_a` high), and rarely tried arms get temporarily boosted (large uncertainty term), so exploration is targeted rather than random.
#### UCB: Q-Value Summary Across Contexts and Categories
UCB was trained separately for each user context with hyperparameter `C ∈ {0.5, 1.0, 2.0}` and `T = 10,000`.

| C | User  | Entertainment | Education | Tech   | Crime  |
|---|---|---:|---:|---:|---:|
| 0.5 | user1 | -8.6959 | 0.8781 | -0.9003 | -3.7810 |
| 0.5 | user2 | 6.4817 | 4.0238 | 0.5960 | 8.1237 |
| 0.5 | user3 | -1.6194 | -10.6160 | 5.7028 | -0.2788 |
| 1.0 | user1 | -6.4213 | 0.8620 | -0.6901 | -4.3808 |
| 1.0 | user2 | 6.6616 | 2.0457 | 2.3175 | 8.1087 |
| 1.0 | user3 | -2.7752 | -8.2252 | 5.7100 | 0.0590 |
| 2.0 | user1 | -8.4426 | 0.8468 | -0.9874 | -3.9613 |
| 2.0 | user2 | 6.4248 | 4.6228 | 3.8698 | 8.1496 |
| 2.0 | user3 | -2.3884 | -10.3110 | 5.7008 | -0.1161 |

#### UCB Inference
1. The best category remains stable across all `C` values:
   - `user1 -> EDUCATION`
   - `user2 -> CRIME`
   - `user3 -> TECH`
2. UCB’s exploration parameter changes the reward estimates of non-optimal arms more than the optimal arm.
3. `user2` shows very strong preference for `CRIME` across all `C` values (highest positive Q).
4. `user1` and `user3` show one clear positive arm with mostly negative alternatives, indicating strong context-specific separation.
5. Hyperparameter sensitivity exists in secondary arms, but policy-level recommendation stays consistent due to stable top arm per context.

![UCB plots](https://github.com/Varchita-Beena/lab3-contextual-bandit/blob/Varchita_D2025PHIL0004/ucb.png)

## Softmax
Softmax was trained separately for each user context with fixed temperature `tau = 1.0` and `T = 10,000`.
$$
P(a) = \frac{e^{Q(a)/\tau}}{\sum_b e^{Q(b)/\tau}}
$$


| Tau | User  | Entertainment | Education | Tech   | Crime  |
|---|---|---:|---:|---:|---:|
| 1.0 | user1 | -8.6169 | 0.8688 | -0.6224 | -3.9681 |
| 1.0 | user2 | 6.0931 | 3.6960 | 2.7839 | 8.1160 |
| 1.0 | user3 | -3.0305 | -9.9000 | 5.7132 | -0.4234 |

#### Softmax Inference

1. Best category per context is stable and matches other algorithms:
   - `user1 -> EDUCATION`
   - `user2 -> CRIME`
   - `user3 -> TECH`
2. Softmax explores proportionally to estimated quality, so clearly weaker arms are sampled less often than in uniform random exploration.
3. `user2` has a strong positive gap for `CRIME`, indicating a clear dominant arm in that context.
4. `user1` and `user3` each show one clear positive arm and mostly negative alternatives, reinforcing context-specific action preference.
5. With fixed `tau=1`, behavior is balanced between exploration and exploitation without explicit epsilon/C tuning.
![UCB plots](https://github.com/Varchita-Beena/lab3-contextual-bandit/blob/Varchita_D2025PHIL0004/softmax.png)

# Recommendation Engine (End-to-End CMAB Inference)

The recommendation pipeline combines user-context prediction with learned bandit policies:

1. Predict user context (`user1`, `user2`, `user3`) from the classifier output (`test_pred.csv`).
2. For the selected policy (Epsilon-Greedy / UCB / Softmax), fetch context-specific `Q` values.
3. Select category greedily using `argmax(Q)`.
4. Filter `news_articles.csv` to required categories only: `ENTERTAINMENT`, `EDUCATION`, `TECH`, `CRIME`.
5. Randomly sample one article from the selected category and return:
   `user_id`, `predicted_context`, `recommended_category`, `link`, `short_description`, `headline`.

## Recommendation Policy Used

- Epsilon-Greedy: `best_eps = 0.05`
- UCB: `best_C = 1.0`
- Softmax: `tau = 1.0` (fixed)

## Learned Context-to-Category Mapping

| Predicted Context | Recommended Category |
|---|---|
| user1 | EDUCATION |
| user2 | CRIME |
| user3 | TECH |

## Recommendation Distribution on Test Set (`n = 2000`)

| Recommended Category | Count |
|---|---:|
| CRIME | 779 |
| EDUCATION | 686 |
| TECH | 535 |

## Inference

1. All three algorithms produced the same top category per context (`user1->EDUCATION`, `user2->CRIME`, `user3->TECH`), so recommendation category choice is consistent across policies.
2. Final recommendation differences across algorithms are mainly in exploration behavior during training, not in final greedy category selection.
3. The engine pipeline: context detection -> category selection -> article-level recommendation.
4. Example outputs confirm article sampling is working (category-correct headlines/links are returned for each user).
