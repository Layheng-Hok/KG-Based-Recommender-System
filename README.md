<div align=center>
   
# Knowledge Graph-Based Recommender System (KGRS)

</div>

This project focuses on building a **Knowledge Graph-Based Recommender System** to enhance user-item interaction predictions. The system utilizes structured knowledge to improve recommendation accuracy, overcoming limitations of traditional approaches.

The README below is just an overview of the project. For details, please refer to the [report](https://github.com/Layheng-Hok/KG-Based-Recommender-System/blob/main/reference/KGRS_Report.pdf).

## Introduction
A recommender system is a type of information filtering system that seeks to predict and show the preferences of a user, offering tailored suggestions for items such as books, movies, or music based on their browsing and selection history. These systems utilize algorithms and data analysis to provide personalized recommendations, enhancing user experience and engagement.

Knowledge Graph (KG)-based recommender system technology uses knowledge graphs as auxiliary information to improve the accuracy and explain-ability of the result of recommendations. Knowledge graphs, which are heterogeneous graphs with nodes representing entities and edges representing relationships, help to illustrate the relationships between items and their attributes, and integrate user and user-side information, capturing the relationships between users and items, as well as user preferences, more accurately.

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
### System Architecture

![kgrs-architecture](https://github.com/user-attachments/assets/9a111a03-425c-4a95-a1e3-221c28c3d872)
### Workflow
1. **Dataset Preparation** – Encode data and construct the knowledge graph.
2. **Model Training** – Optimize the model using training data and minimize the loss function.
3. **Model Testing** – Evaluate performance using AUC and nDCG@5.
4. **Hyperparameter Tuning** – Optimize learning rate, batch size, etc.

### Algorithms Implemented
**Top-K Recommendation Algorithm:** Ranks and filters the top-K most relevant items for each user.

The following pseudocode describes the procedure for evaluating the top-K recommendations for a list of users. For each user, the algorithm identifies the top-K items that they are most likely to interact with, excluding items they have already interacted with in the training dataset. The implementation leverages the scoring function of the model and sorts the items based on their predicted scores.

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

### Experiment Results
#### AUC Evaluation
- **AUC Score:** 0.7003
- Higher embedding dimensions improve performance but may overfit.
- Adjusting margin and negative sampling rate impacts results.

#### nDCG@5 Evaluation
- **nDCG@5 Score:** 0.1844
- Ranking accuracy is influenced by hyperparameter selection.
- Strong correlation between **AUC and nDCG@5**.

## Conclusion
The KG-based recommender system developed in this project demonstrated notable strengths in leveraging structured knowledge to enhance user-item predictions. One key advantage of this approach is its ability to incorporate rich contextual relationships between entities, which allows for more informed and precise recommendations. This contextual understanding surpasses traditional matrix factorization methods that rely solely on interaction data. 

However, several challenges were observed. The computational complexity of the model, particularly during the training phase, posed scalability issues when applied to large-scale datasets. Additionally, the quality of the recommendations heavily depended on the completeness and accuracy of the knowledge graph, which might not always be achievable in practical applications.

Experimental results aligned with expectations, showcasing strong performance on metrics like AUC and nDCG@5. Nonetheless, there were cases where user preferences were underpredicted due to sparsity in the knowledge graph or interaction records. 

Future improvements could focus on integrating graph neural networks (GNNs) to further exploit the structural properties of the knowledge graph. Additionally, fine-tuning hyperparameters, exploring alternative embeddings like TransH or RotatE, and incorporating auxiliary data such as user reviews or temporal information could potentially enhance the model's robustness and accuracy. Despite its limitations, the proposed system lays a solid foundation for knowledge graph-driven recommendation strategies.

## References
- Q. Guo et al., "A Survey on Knowledge Graph-Based Recommender Systems," *IEEE Transactions on Knowledge and Data Engineering*, pp. 1–1, 2020, doi: 10/ghxwqg.
- D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," in *3rd International Conference on Learning Representations, ICLR 2015*, San Diego, CA, USA, May 7-9, 2015. [Online]. Available: [http://arxiv.org/abs/1409.0473](http://arxiv.org/abs/1409.0473)
