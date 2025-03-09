<div align=center>
# Knowledge Graph-Based Recommender System (KGRS)
</div>

This project focuses on building a **Knowledge Graph-Based Recommender System** to enhance user-item interaction predictions. The system utilizes structured knowledge to improve recommendation accuracy, overcoming limitations of traditional approaches.

## Introduction
A recommender system suggests content tailored to users based on past interactions. This project leverages **Knowledge Graphs (KGs)** to model item relationships and enhance recommendations. The primary goal is to develop an algorithm that predicts user interest in unseen items based on structured item attributes and relationships.

## Problem Definition
Given a knowledge graph and a set of interaction records the system predicts user-item interactions:

- **U:** Set of users
- **W:** Set of items
- **t_train^uw:** Binary interaction label (1 = interested, 0 = not interested)
- **f(u, w):** Scoring function predicting user **u**'s interest in item **w**

### Optimization Goals
1. Maximize **AUC (Area Under the Curve)**:

$$
\underset{f}{\max} \ \text{AUC}(f, \mathcal{Y}_{\text{test}})
$$

2. Maximize **nDCG@5 (Normalized Discounted Cumulative Gain at rank 5)**:

$$
\underset{f}{\max} \ nDCG@5(f, \mathcal{Y}_{\text{test}})
$$

## Methodology
### Workflow
1. **Dataset Preparation** – Encode data and construct the knowledge graph.
2. **Model Training** – Optimize the model using training data.
3. **Model Testing** – Evaluate performance using AUC and nDCG@5.
4. **Hyperparameter Tuning** – Optimize learning rate, batch size, etc.

**Algorithms Implemented:** Ranks and filters the top-K most relevant items for each user.

The following algorithm describes the procedure for evaluating the top-K recommendations for a list of users. For each user, the algorithm identifies the top-K items that they are most likely to interact with, excluding items they have already interacted with in the training dataset. The implementation leverages the scoring function of the model and sorts the items based on their predicted scores.

```text
Algorithm: Top-K Recommendation
Input: List of users, number of top recommendations k (default = 5)
Output: List of top-K recommended items for each user

1. Retrieve item list and known positive items for each user
2. Initialize an empty list `sorted_list`
3. For each user in the user list:
   a. Compute head entities
   b. Compute relations
   c. Set tail entities
   d. Compute scores using the model
   e. Sort scores in descending order
   f. Initialize an empty list `sorted_items`
   g. For each index in the sorted scores:
      i. If `sorted_items` has k items, break
      ii. If the user has not interacted with the item, add it to `sorted_items`
   h. Append `sorted_items` to `sorted_list`
4. Return `sorted_list`
```

## Model Analysis
### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 256 |
| Evaluation Batch Size | 1024 |
| Negative Sampling Rate | 1.5 |
| Embedding Dimensions | 16 |
| Margin | 30 |
| Learning Rate | 2e-3 |
| Weight Decay | 5e-4 |
| Epochs | 35 |

### Performance Metrics
- **AUC Score:** 0.7003
- **nDCG@5 Score:** 0.1844

### Complexity Considerations
- Larger KG sizes increase computational cost.
- Higher embedding dimensions improve accuracy but demand more resources.
- Training complexity is **linear** in the number of training samples.

## Experiment Results
### Task 1: AUC Evaluation
- Higher embedding dimensions improve performance but may overfit.
- Adjusting margin and negative sampling rate impacts results.

### Task 2: nDCG@5 Evaluation
- Ranking accuracy is influenced by hyperparameter selection.
- Strong correlation between **AUC and nDCG@5**.

## Conclusion
The KG-based recommender system developed in this project demonstrated notable strengths in leveraging structured knowledge to enhance user-item predictions. One key advantage of this approach is its ability to incorporate rich contextual relationships between entities, which allows for more informed and precise recommendations. This contextual understanding surpasses traditional matrix factorization methods that rely solely on interaction data. 

However, several challenges were observed. The computational complexity of the model, particularly during the training phase, posed scalability issues when applied to large-scale datasets. Additionally, the quality of the recommendations heavily depended on the completeness and accuracy of the knowledge graph, which might not always be achievable in practical applications.

Experimental results aligned with expectations, showcasing strong performance on metrics like AUC and $nDCG@5$. Nonetheless, there were cases where user preferences were underpredicted due to sparsity in the knowledge graph or interaction records. 

Future improvements could focus on integrating graph neural networks (GNNs) to further exploit the structural properties of the knowledge graph. Additionally, fine-tuning hyperparameters, exploring alternative embeddings like TransH or RotatE, and incorporating auxiliary data such as user reviews or temporal information could potentially enhance the model's robustness and accuracy. Despite its limitations, the proposed system lays a solid foundation for knowledge graph-driven recommendation strategies.
