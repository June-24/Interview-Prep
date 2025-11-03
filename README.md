

### Pillar 1: Mathematical Foundations

This is the "why" behind the algorithms. You must be able to explain these concepts intuitively.

#### 1.1: Linear Algebra
* [ ] **Vectors & Matrices:**
    * [ ] Define a vector (geometric, list of numbers).
    * [ ] Define a matrix.
    * [ ] **Operations:** Vector/matrix addition, scalar multiplication.
    * [ ] **Dot Product:** How to calculate it and what it *means* (projection, similarity).
    * [ ] **Matrix Multiplication:** How to do it (shape rules) and *why* (e.g., as a transformation).
    * [ ] **Properties:** Transpose, Identity matrix, Inverse matrix.
* [ ] **Core Concepts:**
    * [ ] **Linear Independence:** What does it mean for vectors to be independent?
    * [ ] **Rank** of a matrix.
    * [ ] **Vector Space / Subspace.**
* [ ] **Advanced (Crucial for ML):**
    * [ ] **Eigenvectors & Eigenvalues:**
        * [ ] Define them: $Av = \lambda v$.
        * [ ] Explain *intuitively* (vectors that only stretch/shrink).
        * [ ] How to find them (conceptually).
        * [ ] Relate this directly to **PCA**.
    * [ ] **Singular Value Decomposition (SVD):**
        * [ ] What it decomposes a matrix into ($U\Sigma V^T$).
        * [ ] Explain its relationship to PCA and its use in dimensionality reduction / matrix factorization (e.g., recommender systems).

#### 1.2: Calculus
* [ ] **Derivatives:**
    * [ ] Define a derivative (slope, rate of change).
    * [ ] **Partial Derivatives:** Taking a derivative with respect to one variable.
    * [ ] **The Gradient:**
        * [ ] Define it (a vector of all partial derivatives).
        * [ ] Explain its *meaning* (direction of steepest ascent).
* [ ] **The Chain Rule:**
    * [ ] State the rule: $f(g(x))' = f'(g(x))g'(x)$.
    * [ ] **This is the single most important concept in deep learning.**
    * [ ] Be able to walk through a simple 2-layer neural network calculation and explain how the chain rule is used in **Backpropagation**.
* [ ] **Optimization:**
    * [ ] **Gradient Descent:**
        * [ ] Explain the algorithm (take a step in the *opposite* direction of the gradient).
        * [ ] Write the update rule: $w_{\text{new}} = w_{\text{old}} - \alpha \nabla J(w)$.
        * [ ] Define **Learning Rate ($\alpha$)** and what happens if it's too high or too low.
    * [ ] Explain **Convexity** (why it's great, e.g., for linear regression) vs. Non-Convexity (for neural networks).
    * [ ] Define **Local vs. Global Minima.**

---

### Pillar 2: Probability & Statistics

This is the "how" you handle uncertainty and evaluate results.

#### 2.1: Foundational Probability
* [ ] **Core Concepts:**
    * [ ] Define **Sample Space, Event, Probability.**
    * [ ] **Conditional Probability:** $P(A|B) = P(A \cap B) / P(B)$. Be able to explain this with a 2x2 table (e.g., "given a user clicked, what's the chance they buy?").
    * [ ] **Independence:** $P(A \cap B) = P(A)P(B)$.
    * [ ] **Joint Probability** vs. **Marginal Probability.**
* [ ] **Bayes' Theorem:**
    * [ ] Write the formula: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$.
    * [ ] Define each term: **Prior ($P(A)$), Likelihood ($P(B|A)$), Posterior ($P(A|B)$), Evidence ($P(B)$).**
    * [ ] Explain its relevance (e.g., Naive Bayes, A/B testing, Bayesian inference).
* [ ] **Random Variables:**
    * [ ] Define **Discrete** vs. **Continuous** variables.
    * [ ] **Distributions (Know the stories):**
        * [ ] **Uniform:** (All outcomes equally likely).
        * [ ] **Bernoulli:** (One trial, success/fail).
        * [ ] **Binomial:** (N trials, k successes).
        * [ ] **Poisson:** (Event rate in a time period).
        * [ ] **Normal (Gaussian):** (The bell curve, Central Limit Theorem).
    * [ ] Define **PDF** (Probability Density Function) and **CDF** (Cumulative Density Function).

#### 2.2: Foundational Statistics
* [ ] **Descriptive Statistics:**
    * [ ] **Central Tendency:** Mean, Median, Mode (and when to use each).
    * [ ] **Variability:** Variance, Standard Deviation, Range, Interquartile Range (IQR).
* [ ] **Inferential Statistics:**
    * [ ] **Central Limit Theorem (CLT):** Explain it *intuitively* (means of samples become normally distributed).
    * [ ] **Hypothesis Testing:**
        * [ ] Define **Null Hypothesis ($H_0$)** and **Alternative Hypothesis ($H_1$).**
        * [ ] Define **p-value** (the probability of observing your data, or more extreme, *assuming the null is true*).
        * [ ] Define **Significance Level ($\alpha$)** (e.g., 0.05).
        * [ ] Define **Type I Error (False Positive)** and **Type II Error (False Negative).**
    * [ ] **A/B Testing:** Be able to design an A/B test (e.g., for a website button).
* [ ] **Key ML Concepts:**
    * [ ] **Maximum Likelihood Estimation (MLE):** Explain it (finding parameters that make the *observed data* most probable).
    * [ ] **Maximum A Posteriori (MAP):** Explain it (like MLE, but with a *prior belief* about the parameters).

---

### Pillar 3: CS, Programming & Data Handling

This is the "how" you actually build it.

#### 3.1: Data Structures & Algorithms (The "FAANG" part)
* [ ] **Big O Notation:**
    * [ ] Define $O(1)$, $O(\log n)$, $O(n)$, $O(n \log n)$, $O(n^2)$, $O(2^n)$.
    * [ ] Be able to analyze the time and space complexity of an algorithm.
* [ ] **Data Structures:**
    * [ ] **Array / List:** (Pros/cons).
    * [ ] **Hash Map (Dictionary):** (Pros/cons, $O(1)$ lookup). This is the *most important* data structure.
    * [ ] **Tree / Binary Search Tree:** (Operations, pros/cons).
    * [ ] **Graph:** (Representations: Adjacency Matrix vs. List).
* [ ] **Algorithms:**
    * [ ] **Sorting:** Merge Sort, Quick Sort (how they work).
    * [ ] **Searching:** Binary Search.
    * [ ] **Graph Traversal:** Breadth-First Search (BFS), Depth-First Search (DFS).

#### 3.2: Python for Data Science
* [ ] **NumPy:**
    * [ ] `ndarray` object.
    * [ ] **Vectorization:** Why it's faster than `for` loops.
    * [ ] **Broadcasting:** What it is, how it works.
    * [ ] Indexing (slicing, boolean indexing).
* [ ] **Pandas:**
    * [ ] `DataFrame` and `Series` objects.
    * [ ] Reading data (`read_csv`).
    * [ ] Indexing (`.loc`, `.iloc`).
    * [ ] **`groupby`:** Be a master of this (split-apply-combine).
    * [ ] **Merging/Joining:** `merge`, `join`, `concat`.
    * [ ] Handling Missing Data: `dropna`, `fillna`.
    * [ ] `apply`, `map`, `applymap`.

#### 3.3: SQL
* [ ] **Basic Queries:** `SELECT`, `FROM`, `WHERE`, `LIMIT`.
* [ ] **Aggregation:** `GROUP BY`, `COUNT`, `SUM`, `AVG`, `MAX`.
* [ ] **Filtering Aggregates:** `HAVING`.
* [ ] **Joins:** `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN`.
* [ ] **Subqueries:** (Queries inside a `WHERE` or `FROM` clause).
* [ ] **Window Functions:** (Advanced, but impressive): `ROW_NUMBER`, `RANK`, `PARTITION BY`.

---

### Pillar 4: "Classic" Machine Learning Models

This is your core toolkit. For *each* model, know:
1.  **How it works** (the main intuition).
2.  **Pros & Cons.**
3.  **Key Hyperparameters.**
4.  **Assumptions** it makes.
5.  **When to use it.**

#### 4.1: Supervised Learning (Regression)
* [ ] **Linear Regression:**
    * [ ] Equation: $y = \beta_0 + \beta_1x_1 + ... + \epsilon$.
    * [ ] **Cost Function:** Mean Squared Error (MSE).
    * [ ] **Optimization:** Gradient Descent (or Normal Equation).
    * [ ] **Assumptions:** Linearity, Independence, Homoscedasticity, Normality of residuals (L.I.N.E.).
* [ ] **Regularization (L1 & L2):**
    * [ ] **The Problem:** Overfitting.
    * [ ] **Ridge Regression (L2):**
        * [ ] Adds $\lambda\sum(w_i^2)$ to the cost function.
        * [ ] *Effect:* Shrinks weights, but doesn't make them zero.
    * [ ] **Lasso Regression (L1):**
        * [ ] Adds $\lambda\sum|w_i|$ to the cost function.
        * [ ] *Effect:* Can make weights *exactly zero* (performs feature selection).
    * [ ] **Elastic Net:** (A combination of L1 and L2).

#### 4.2: Supervised Learning (Classification)
* [ ] **Logistic Regression:**
    * [ ] **Sigmoid Function:** How it maps output to (0, 1).
    * [ ] **Cost Function:** Log Loss / Binary Cross-Entropy.
    * [ ] **Decision Boundary:** (Linear).
    * [ ] How it relates to Linear Regression.
* [ ] **k-Nearest Neighbors (k-NN):**
    * [ ] How it works (lazy learner, "votes" from neighbors).
    * [ ] Hyperparameter: `k`.
    * [ ] Pros (simple, non-parametric) & Cons (slow, curse of dimensionality).
    * [ ] Importance of **Feature Scaling**.
* [ ] **Support Vector Machines (SVM):**
    * [ ] Intuition: "Maximum Margin Classifier."
    * [ ] Define: Support Vectors, Margin, Hyperplane.
    * [ ] **The Kernel Trick:**
        * [ ] What it is (mapping to a higher dimension).
        * [ ] Common kernels: Linear, Polynomial, **RBF (Radial Basis Function)**.
    * [ ] Hyperparameters: `C` (regularization) and `gamma` (for RBF).
* [ ] **Naive Bayes:**
    * [ ] How it works (uses Bayes' Theorem).
    * [ ] The "Naive" Assumption: **Conditional Independence** of features.
    * [ ] Pros (fast, good for text) & Cons (the assumption is often wrong).
    * [ ] Types: Gaussian, Multinomial.

#### 4.3: Tree-Based Models (HIGH IMPORTANCE)
* [ ] **Decision Trees:**
    * [ ] How they are built (recursive splitting).
    * [ ] **Splitting Criteria:**
        * [ ] **Gini Impurity:** (Used by CART).
        * [ ] **Entropy / Information Gain:** (Used by ID3, C4.5).
    * [ ] How to stop splitting (e.g., `max_depth`, `min_samples_leaf`).
    * [ ] **Overfitting:** Why they are "high variance" models.
    * [ ] **Pruning.**
* [ ] **Ensemble Methods (The core idea):**
    * [ ] Explain **Bagging (Bootstrap Aggregating):**
        * [ ] How it works (train N models on N bootstrapped samples).
        * [ ] *Result:* Reduces **variance**.
    * [ ] Explain **Boosting:**
        * [ ] How it works (train N models *sequentially*, each correcting the last's errors).
        * [ ] *Result:* Reduces **bias** (and variance).
* [ ] **Random Forest:**
    * [ ] It is **Bagging** of Decision Trees.
    * [ ] *Two* sources of randomness: Bootstrapped samples + random subset of *features* at each split.
    * [ ] Pros (robust, handles non-linear data) & Cons (less interpretable).
* [ ] **Gradient Boosting Machines (GBM):**
    * [ ] It is **Boosting** where each new tree predicts the **residual (error)** of the previous ensemble.
    * [ ] **XGBoost / LightGBM:**
        * [ ] Know *why* they are better than standard GBM (regularization, speed, handling missing values, parallelization).

#### 4.4: Unsupervised Learning
* [ ] **K-Means Clustering:**
    * [ ] The algorithm (Initialize, Assign, Update).
    * [ ] Hyperparameter: `k`.
    * [ ] **Choosing k:** Elbow Method, Silhouette Score.
    * [ ] Cons (sensitive to initialization, assumes spherical clusters).
* [ ] **Hierarchical Clustering:**
    * [ ] How it works (Agglomerative).
    * [S] How to read a **Dendrogram**.
* [ ] **DBSCAN:**
    * [ ] How it works (density-based).
    * [ ] Pros (can find arbitrary shapes, handles outliers).
* [ ] **Dimensionality Reduction (PCA):**
    * [ ] **Principal Component Analysis (PCA):**
        * [ ] **The Goal:** Find new axes (principal components) that maximize variance.
        * [ ] **The Method:** (Uses Eigenvectors of the covariance matrix).
        * [ ] **How to use it:** Scale data -> Fit -> Choose `n_components` (e.g., via explained variance plot).

---

### Pillar 5: Model Evaluation & The ML Lifecycle


#### 5.1: Data Preprocessing
* [ ] **Feature Scaling:**
    * [ ] **StandardScaler:** (z-score, mean=0, std=1). *When:* When data is Gaussian.
    * [ ] **MinMaxScaler:** (to [0, 1]). *When:* For non-Gaussian data, or for NNs.
    * [ ] *Why:* For algorithms sensitive to distance (k-NN, SVM, PCA, NNs).
* [ ] **Categorical Encoding:**
    * [ ] **One-Hot Encoding:** (Pros/cons).
    * [ ] **Label Encoding:** (Pros/cons).
    * [ ] **Target Encoding:** (What it is, risk of target leakage).
* [ ] **Handling Missing Data:**
    * [ ] `dropna`.
    * [ ] **Imputation:** Mean/Median/Mode, k-NN Imputation, Model-based Imputation.
* [ ] **Feature Engineering:**
    * [ ] Binning, Interaction features, Polynomial features.

#### 5.2: Model Validation
* [ ] **The Bias-Variance Trade-off:**
    * [ ] Define **Bias** (error from wrong assumptions, underfitting).
    * [ ] Define **Variance** (error from sensitivity to training data, overfitting).
    * [ ] Be able to draw the graph (U-shaped total error).
    * [ ] Relate this to model complexity (e.g., high-degree poly, deep tree).
* [ ] **Overfitting vs. Underfitting:**
    * [ ] How to detect (Train error << Test error).
    * [ ] How to fix Overfitting (more data, regularization, less complexity).
    * [ ] How to fix Underfitting (more complex model, new features).
* [ ] **Cross-Validation:**
    * [ ] Why we do it (better estimate of test error).
    * [ ] **K-Fold Cross-Validation:** (The algorithm).
    * [ ] **Stratified K-Fold:** (Why it's needed for imbalanced data).
    * [ ] **Data Leakage:** What it is, and how to avoid it (e.g., scale *inside* the CV loop).

#### 5.3: Evaluation Metrics (CRITICAL)
* [ ] **Regression Metrics:**
    * [ ] **MSE / RMSE:** (Pros/cons).
    * [ ] **MAE:** (Pros: robust to outliers).
    * [ ] **R-squared:** (What it means: "proportion of variance explained").
* [ ] **Classification Metrics:**
    * [ ] **The Confusion Matrix:**
        * [ ] Define **True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN).**
    * [ ] **Accuracy:** (TP+TN) / Total. *When it's a bad metric* (imbalanced data).
    * [ ] **Precision:** TP / (TP+FP). ("Of all positive predictions, how many were right?").
    * [ ] **Recall:** TP / (TP+FN). ("Of all actual positives, how many did we find?").
    * [ ] **The Precision-Recall Trade-off:** Be able to explain it.
    * [ ] **F1-Score:** (Harmonic mean of Precision and Recall).
    * [ ] **ROC Curve & AUC:**
        * [ ] **ROC:** (True Positive Rate vs. False Positive Rate).
        * [ ] **AUC:** (Area Under the Curve). What an AUC of 0.5 means (random).
    * [ ] **PR Curve:** (Precision vs. Recall). When to use this over ROC (imbalanced data).
* [ ] **Handling Imbalanced Data:**
    * [ ] **Metrics:** (Use Precision, Recall, F1, AUC-PR).
    * [ ] **Resampling Techniques:**
        * [ ] **SMOTE** (Synthetic Minority Over-sampling TEchnique).
        * [ ] Undersampling.
    * [ ] **Class Weights:** (How to adjust the model's cost function).

---

### Pillar 6: Deep Learning Foundations

The path to Transformers.

#### 6.1: Neural Network Basics
* [ ] **The Perceptron / MLP:**
    * [ ] Draw a simple Multi-Layer Perceptron (Input, Hidden, Output layers).
    * [ ] **Activation Functions:**
        * [ ] **Sigmoid:** (Pros/cons: saturates, vanishing gradient).
        * [ ] **Tanh:** (Pros/cons: zero-centered, but still saturates).
        * [ ] **ReLU (Rectified Linear Unit):**
            * [ ] The function: $max(0, x)$.
            * [ ] **Pros:** Fast, non-saturating, default choice.
            * [ ] **Cons:** "Dying ReLU" problem.
        * [ ] **Leaky ReLU:** (Solves the dying ReLU problem).
        * [ ] **Softmax:** (How it works, output is a probability distribution).
* [ ] **Training NNs:**
    * [ ] **Cost Function:** **Cross-Entropy Loss** (for classification).
    * [ ] **Backpropagation:**
        * [ ] Explain it *intuitively* (chain rule, propagating error gradient backward).
    * [ ] **Gradient Descent Variants:**
        * [ ] **Stochastic (SGD):** (1 sample).
        * [ ] **Mini-Batch:** (The standard).
        * [ ] **Batch:** (The whole dataset).
    * [ ] **Optimizers:**
        * [ ] **Momentum:** (Adds "mass" to the gradient).
        * [ ] **Adam:** (Combines Momentum + RMSProp, the default choice).
* [ ] **Regularization for NNs:**
    * [ ] **Dropout:**
        * [ ] How it works (randomly zero-ing out neurons during *training*).
        * [ ] Why it works (prevents co-adaptation, like an ensemble).
    * [ ] **Batch Normalization:**
        * [ ] How it works (standardizes activations *inside* the network).
        * [ ] **Pros:** Speeds up training, allows higher learning rates, adds regularization.
    * [ ] **Early Stopping.**
    * [ ] **Data Augmentation.**

---

### Pillar 7: Advanced Architectures (The "GPT" Stuff)

This is the modern state-of-the-art.

#### 7.1: Computer Vision (CNNs)
* [ ] **Convolutional Neural Networks (CNNs):**
    * [ ] **Convolutional Layer:**
        * [ ] Define: **Kernel / Filter, Padding, Stride.**
        * [S] Explain *why* it works (parameter sharing, spatial hierarchy).
    * [ ] **Pooling Layer:**
        * [ ] **Max Pooling** vs. **Average Pooling.**
        * [ ] *Purpose:* Down-sampling, translation invariance.
    * [ ] **Architecture:** (Conv -> ReLU -> Pool -> ... -> Flatten -> Dense).
* [ ] **Key Architectures (Know the "one big idea"):**
    * [ ] **LeNet / AlexNet:** (The pioneers).
    * [ ] **VGG:** (Showed that *depth* with simple 3x3 filters works).
    * [ ] **ResNet:**
        * [ ] **The Problem:** Vanishing gradients in very deep nets.
        * [ ] **The Solution:** **Residual / Skip Connections** ($x + F(x)$).
* [ ] **Transfer Learning:**
    * [ ] What it is (using a pre-trained backbone).
    * [ ] How to do it (freeze early layers, re-train the final "head").

#### 7.2: Sequential Models (RNNs & LSTMs)
* [ ] **"Classic" NLP:**
    * [ ] **Bag-of-Words (BoW):** (What it is, pros/cons).
    * [ ] **TF-IDF:** (What it is, why it's better than BoW).
    * [ ] **Word Embeddings (Word2Vec):**
        * [ ] The *idea*: Represent words as dense vectors.
        * [ ] **Skip-gram** vs. **CBOW.**
        * [ ] "King - Man + Woman = Queen".
* [ ] **Recurrent Neural Networks (RNNs):**
    * [ ] How they work (a loop, passing hidden state $h_t$).
    * [ ] **The Problem:** **Vanishing & Exploding Gradients** (short-term memory).
* [ ] **LSTM & GRU:**
    * [ ] **LSTM (Long Short-Term Memory):**
        * [ ] The *idea*: A "cell state" (conveyor belt) and **gates**.
        * [ ] Name the gates: **Forget, Input, Output.**
    * [ ] **GRU:** (A simplified LSTM).
* [ ] **Seq2Seq:**
    * [ ] The **Encoder-Decoder** architecture.
    * [ ] The "context vector" bottleneck.

#### 7.3: The Transformer Era (Your "GPT-2" goal)
* [ ] **The "Attention is All You Need" Paper:**
    * [ ] **The Core Problem with RNNs:** Sequential, can't parallelize, long-range dependencies are hard.
* [ ] **Self-Attention Mechanism:**
    * [ ] **Query (Q), Key (K), Value (V):**
        * [ ] Be able to explain this *intuitively* (Query: "what I'm looking for", Key: "what I have", Value: "what I'll give you").
    * [ ] The equation: $Attention(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
    * [ ] **Multi-Head Attention:** (Running attention in parallel "subspaces").
* [ ] **The Transformer Architecture:**
    * [ ] **Positional Encodings:** (How it knows *where* words are).
    * [ ] **Encoder Block:** (Self-Attention -> Add&Norm -> FeedForward -> Add&Norm).
    * [ ] **Decoder Block:** (Masked Self-Attention -> Cross-Attention -> ...).
* [ ] **Key Models (Know the DIFFERENCE):**
    * [ ] **BERT (Encoder-Only):**
        * [ ] **Pre-training:** **Masked Language Model (MLM)** and Next Sentence Prediction.
        * [ ] *Property:* Deeply bidirectional. Good for *analysis* tasks (sentiment, NER).
    * [ ] **GPT (Decoder-Only):**
        * [ ] **Pre-training:** **Causal Language Model (CLM)** (predict the next word).
        * [ ] *Property:* Autoregressive. Good for *generation* tasks.
    * [ ] **T5 (Encoder-Decoder):** (Text-to-text framework).
* [ ] **Modern Concepts:**
    * [ ] **Fine-Tuning:** (How to adapt a pre-trained model).
    * [ ] **Prompt Engineering.**
    * [ ] **RAG (Retrieval-Augmented Generation):** (Combining a language model with a knowledge base).

---

### Pillar 8: MLOps & System Design


#### 8.1: The ML Lifecycle
* [ ] **Data Ingestion:** (APIs, databases, data lakes).
* [ ] **Data Validation:** (Schema checks, drift detection, TFDV).
* [ ] **Feature Stores:** (What they are, why they are useful).
* [ ] **Model Training & Experiment Tracking:** (Tools: **MLflow**, Weights & Biases).
* [ ] **Model Versioning:** (Git for code, **DVC** for data).
* [ ] **Model Serving:**
    * [ ] **Batch Prediction** vs. **Real-Time Inference (API).**
    * [ ] **Docker:** (Containerization, what it is).
    * [ ] **Kubernetes:** (Orchestration, at a high level).
* [ ] **Model Monitoring:**
    * [ ] **Data Drift:** (Input data distribution changes).
    * [ ] **Concept Drift:** (Relationship between input and output changes).
    * [ ] **Feedback Loops:** (How to retrain your model).

#### 8.2: ML System Design
* [ ] Be ready for a question like: **"Design a YouTube recommendation system."** or **"Design a spam filter."**
* [ ] **Your Checklist:**
    * [ ] **1. Requirements & Scope:** (Clarify: Real-time? Scale? What is the *real* goal?).
    * [ ] **2. Metrics:** (Offline: Precision@k, Recall, NDCG. Online: Click-Through Rate, Watch Time).
    * [ ] **3. Data:** (What data to collect? User history, video metadata, context).
    * [ ] **4. Architecture (High Level):** (Data pipeline, training, serving).
    * [ ] **5. Model Selection:**
        * [ ] **Candidate Generation:** (Fast model, e.g., matrix factorization, to get 1000 candidates).
        * [ ] **Ranking:** (Slower, more complex model, e.g., a deep NN, to rank the top 1000).
    * [ ] **6. Serving & Monitoring:** (How to serve, how to monitor for drift, how to retrain).

---

### 🗣️ Pillar 9: The Interview Itself

#### 9.1: Behavioral
* [ ] Have 3-5 projects ready to discuss using the **STAR method:**
    * [ ] **S**ituation: What was the problem/context?
    * [ ] **T**ask: What was your specific role?
    * [ ] **A**ction: What did you *do*? (Be technical. "I engineered new features by..." "I compared 3 models by...").
    * [ ] **R**esult: What was the outcome? (Quantify it! "Improved F1 by 10%..." "Reduced customer churn...").
* [ ] **Be ready for:**
    * [ ] "Tell me about a time you failed."
    * [ ] "Tell me about a conflict with a teammate."
    * [ ] "Why [This Company]?"
    * [ ] "Why ML?"




### Pillar 10 : Deep Learning Foundations (Expanded)

#### 10.2: Automatic Differentiation (The "Autodiff" Stuff)
This is the engine *under* backpropagation. PyTorch calls it `autograd`, TensorFlow calls it `GradientTape`. You must know *how* it works.

* [ ] **The Core Idea:**
    * [ ] Define **Automatic Differentiation (Autodiff):** A technique to numerically compute the derivative of a function (your loss function) with respect to its parameters (your model weights).
    * [ ] Explain the **Computational Graph:** How any complex function (like a neural net) can be broken down into a graph of simple, elementary operations (add, multiply, `exp`, `max`, etc.).
    * [ ] **Key difference:** Autodiff is *not* symbolic differentiation (like in Mathematica, which is slow) and *not* finite differences (like $\frac{f(x+h) - f(x)}{h}$, which is imprecise).
* [ ] **Modes of Autodiff (CRITICAL):**
    * [ ] **Forward-Mode Autodiff:**
        * *Intuition:* Calculates how *one* input affects *all* nodes as it flows *forward*.
        * *Analogy:* You ask "If I change weight $w_1$ by 1, how does the loss change?"
        * *Efficiency:* Good for functions with *few inputs* and *many outputs* ($R^n \rightarrow R^m$ where $n \ll m$). This is **not** the typical ML case.
    * [ ] **Reverse-Mode Autodiff (This is Backpropagation):**
        * *Intuition:* Calculates how *all* inputs affect *one* output (the loss) as it flows *backward*.
        * *Analogy:* You ask "To change the final loss, how much should *every* weight in the network contribute?"
        * *Efficiency:* Good for functions with *many inputs* (all model weights) and *one output* (the final loss scalar). This **is** the ML case.
        * [ ] **This is why we use it:** It's a single backward pass to find the gradient of the loss with respect to *millions* of parameters.

#### 10.3: Non-Linearity Functions (Expanded Activation Checklist)
You need these to escape simple linear combinations. For each, know its formula, graph shape, and pros/cons.

* [ ] **The "Classic" Functions:**
    * [ ] **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
        * *Pros:* Squashes output to (0, 1), good for *binary classification output*.
        * *Cons:* **1. Vanishing Gradients:** Flattens at ends, gradients $\rightarrow 0$. **2. Not Zero-Centered:** Output is always positive, can make optimization tricky.
    * [ ] **Tanh (Hyperbolic Tangent):** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
        * *Pros:* Squashes to (-1, 1), **Zero-Centered** (better than sigmoid).
        * *Cons:* Still has the **Vanishing Gradient** problem.
* [ ] **The "Modern" (Default) Functions:**
    * [ ] **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$
        * *Pros:* **1. Solves Vanishing Gradient** (for $x>0$). **2. Computationally Fast:** It's just a `max` operation. **3. Induces Sparsity.**
        * *Cons:* **1. "Dying ReLU" Problem:** Neurons can get "stuck" at $0$ if $x<0$ and never recover. **2. Not Zero-Centered.**
    * [ ] **Leaky ReLU:** $f(x) = \max(0.01x, x)$
        * *Pros:* **Solves the Dying ReLU problem** by allowing a small, non-zero gradient for negative inputs.
        * *Cons:* The slope `0.01` is a "magic number."
    * [ ] **PReLU (Parametric ReLU):** $f(x) = \max(\alpha x, x)$
        * *Pros:* Like Leaky ReLU, but $\alpha$ is a **learnable parameter**, not a fixed value.
    * [N] **ELU (Exponential Linear Unit):**
        * *Pros:* Zero-centered output, solves dying ReLU, smooth curve (can be more robust).
        * *Cons:* Slower computation (due to $e^x$).
* [ ] **The Output Layer Functions:**
    * [ ] **Softmax:** $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$
        * *Purpose:* Turns a vector of raw scores (logits) into a probability distribution that sums to 1.
        * *Use Case:* **Multi-class classification**.

---

### Pillar 11: More "Classic" ML Models

Expanding the toolkit for specific data types and problems.

* [ ] **Discriminative Models (LDA/QDA):**
    * [ ] **Linear Discriminant Analysis (LDA):**
        * *Intuition:* A classifier that finds a linear boundary by modeling the distribution of each class (as a Gaussian).
        * *Assumption:* **Assumes all classes share the same covariance matrix.**
        * *Use Case:* Dimensionality reduction *before* classification.
    * [ ] **Quadratic Discriminant Analysis (QDA):**
        * *Intuition:* Same as LDA, but allows for curved, non-linear boundaries.
        * *Assumption:* **Each class has its *own* covariance matrix.**
        * *Trade-off:* More flexible than LDA, but has more parameters and can overfit.
* [ ] **Generative Probabilistic Models:**
    * [ ] **Gaussian Mixture Models (GMM):**
        * *Intuition:* A "soft clustering" algorithm. Assumes data is a mix of several Gaussian distributions.
        * *Algorithm:* **Expectation-Maximization (E-M)**.
            * *E-Step:* Calculates the probability that each point belongs to each cluster.
            * *M-Step:* Updates the mean, variance, and weight of each cluster based on those probabilities.
        * *vs. K-Means:* GMMs are probabilistic (soft assignment) and can find non-spherical (elliptical) clusters.
    * [ ] **Hidden Markov Models (HMM):**
        * *Intuition:* A model for sequential data where the system is in a "hidden" state that you can't see, but it emits "observable" outputs.
        * *Components:* States, Observation Probabilities, Transition Probabilities.
        * *Use Case:* "Classic" NLP (Part-of-Speech tagging), speech recognition, bioinformatics.

---

### Pillar 12: Advanced Architectures (Expanded)

This is the modern deep learning landscape beyond basics.

#### 7.4: Generative & Unsupervised Architectures
* [ ] **Autoencoders (AE):**
    * *Architecture:* A "bottleneck" network: **Encoder** (compresses data) $\rightarrow$ **Latent Space** (the compressed code) $\rightarrow$ **Decoder** (reconstructs data).
    * *Purpose:* Unsupervised dimensionality reduction, feature learning, denoising.
    * *Cost Function:* **Reconstruction Loss** (e.g., MSE).
    * *Types:* Denoising AE, Sparse AE, Undercomplete AE.
* [ ] **Variational Autoencoders (VAE):**
    * *vs. AE:* A VAE is **generative**. The latent space isn't just a point, it's a *probability distribution* (a mean and variance).
    * *How it works:* The decoder samples from this learned distribution to generate *new* data.
    * *Cost Function:* Reconstruction Loss + **Kullback-Leibler (KL) Divergence** (this acts as a regularizer, forcing the latent space to be smooth and continuous).
* [ ] **Generative Adversarial Networks (GANs):**
    * *Intuition:* A "two-player game" to generate realistic data.
    * [ ] **The Generator ($G$):** The "Forger." Tries to create fake data (e.g., images) from random noise.
    * [ ] **The Discriminator ($D$):** The "Detective." Tries to tell the difference between real data and the Generator's fake data.
    * *Training Process (Minimax Game):*
        1.  Train $D$ on real and fake data (so it gets better at detecting fakes).
        2.  Freeze $D$, and train $G$ to *fool* $D$ (make $D$ label its fakes as "real").
    * *Challenges:* Mode collapse, unstable training.
* [ ] **Graph Neural Networks (GNNs):**
    * *What:* A class of neural networks for data structured as a **graph** (nodes and edges).
    * *Intuition:* They work via "message passing." Each node updates its own state (embedding) by aggregating information from its neighbors.
    * *Use Cases:* Social network analysis, recommendation (user-item graphs), molecular chemistry (molecule graphs).

---

### Pillar 13: Natural Language Processing (Deep Dive)

This is its own universe. We'll split it into "Classic" and "Modern."

#### 13.1: The "Classic" NLP Pipeline
This is the pre-Transformer foundation. You must know these steps.

* [ ] **Text Preprocessing:**
    * [ ] **Sentence Segmentation:** Splitting text into individual sentences.
    * [ ] **Tokenization:** Splitting sentences into individual words/tokens. (Be able to discuss challenges: "don't", "N.Y.", "U.S.A.").
    * [ ] **Stop Word Removal:** Removing common words ("the", "is", "a").
    * [ ] **Stemming:**
        * *What:* Chopping words to their root. (e.g., "running", "runs" $\rightarrow$ "run").
        * *Algorithm:* Porter Stemmer.
        * *Con:* Can create non-words (e.g., "studies" $\rightarrow$ "studi").
    * [ ] **Lemmatization:**
        * *What:* Reducing words to their dictionary form (lemma). (e.g., "was" $\rightarrow$ "be", "better" $\rightarrow$ "good").
        * *Pro:* More accurate than stemming. *Con:* Slower, needs a dictionary.
* [ ] **Text Representation (Feature Engineering):**
    * [ ] **Bag-of-Words (BoW):**
        * *What:* Representing a document as a vector of word counts.
        * *Pro:* Simple. *Con:* Loses all word order and context.
    * [ ] **TF-IDF (Term Frequency - Inverse Document Frequency):**
        * *What:* A "smarter" BoW. It's $TF \times IDF$.
        * [ ] **TF (Term Frequency):** (How often a word appears in *this* doc).
        * [ ] **IDF (Inverse Doc Freq):** (How rare a word is across *all* docs).
        * *Intuition:* Gives high weight to words that are *frequent in this doc* but *rare everywhere else*.
    * [ ] **N-Grams:** (e.g., "New York", "I am happy"). Captures *some* local word order.
* [ ] **Classic NLP Models:**
    * [ ] Naive Bayes (for text classification/spam).
    * [ ] SVM (very strong for text classification).
    * [ ] HMMs (for Part-of-Speech Tagging, Named Entity Recognition).

#### 13.2: The "Modern" NLP Pipeline (The Transformer Era)
This is the "GPT-2" stuff.

* [ ] **Word Embeddings (The Precursor):**
    * [ ] **Word2Vec:**
        * *Intuition:* Represent words as dense vectors. "You shall know a word by the company it keeps."
        * *Architectures:* **CBOW** (predict center word from context) vs. **Skip-Gram** (predict context from center word).
    * [ ] **GloVe:** (Global Vectors, uses matrix factorization on co-occurrence matrix).
    * [ ] **The Problem they Solve:** *Context-free.* The vector for "bank" (river) is the same as "bank" (money).
* [ ] **The Transformer Architecture (The "Attention is All You Need" paper):**
    * [ ] **Core Idea:** Replaced RNNs/LSTMs with **Self-Attention** to handle long-range dependencies and allow parallelization.
    * [ ] **Self-Attention (The core mechanism):**
        * [ ] **Query ($Q$):** "What I'm looking for."
        * [ ] **Key ($K$):** "What I have."
        * [ ] **Value ($V$):** "What I'll give you."
        * [ ] *Intuition:* For each word, $Q$ "looks at" all other words' $K$s to find a similarity score. That score weights the $V$s. It *learns* the relationships between all word-pairs.
    * [ ] **Multi-Head Attention:** Running self-attention multiple times in parallel to capture different types of relationships.
    * [ ] **Positional Encodings:** Since the model has no RNN, it needs these vectors to tell it *where* in the sentence a word is.
* [ ] **The Great Divide: Key Architectures (CRITICAL):**
    * [ ] **BERT (Encoder-Only):**
        * *Stands For:* Bidirectional Encoder Representations from Transformers.
        * *Pre-training Task:* **Masked Language Model (MLM)**. (It "masks" 15% of words and tries to predict them from *both* left and right context).
        * *Property:* **Deeply Bidirectional.** It sees the whole sentence at once.
        * *Use Case:* **Analysis & Understanding** (Sentiment, NER, Classification).
    * [ ] **GPT-2 (Decoder-Only):**
        * *Stands For:* Generative Pre-trained Transformer.
        * *Pre-training Task:* **Causal Language Model (CLM)**. (Predict the *next* word, given all previous words).
        * *Property:* **Autoregressive & Unidirectional.** It only looks to the left.
        * *Use Case:* **Generation** (Writing text, chatbots).
    * [ ] **T5 (Encoder-Decoder):**
        * *Stands For:* Text-to-Text Transfer Transformer.
        * *Intuition:* Treats *every* NLP task as a text-to-text problem (e.g., for translation, input is "translate English to German: ...").




### The Only Way to Answer: The STAR Method

Before the list, remember the **STAR method**. Every answer must follow this structure:

* **S (Situation):** Set the scene. What was the project? Who was involved? (1-2 sentences)
* **T (Task):** What was *your* specific responsibility in this situation? (1 sentence)
* **A (Action):** What did you *do*? Detail the steps you took. This is the main part of your answer. Use "I," not "we."
* **R (Result):** What was the outcome? **Quantify it.** (e.g., "We improved model accuracy by 5%," "Reduced API latency by 300ms," "We launched the feature on time.")

---

### 1. Project & Technical Execution

*(How you get work done, from start to finish)*

1.  Tell me about the most complex ML project you've worked on.
2.  Describe a project you are most proud of. What was your role?
3.  Walk me through a project from conception to deployment.
4.  Tell me about a time you had to make a significant technical decision (e.g., choice of model, data source, framework).
5.  How did you weigh the trade-offs?
6.  Describe a project where you had to use a large or messy dataset. How did you handle it?
7.  Tell me about a time you had to balance model performance with inference speed or cost.
8.  What's the most innovative or creative technical solution you've ever built?
9.  Walk me through a time you had to build a system with high scalability in mind.
10. Tell me about a time you owned a project end-to-end.
11. Describe a time you had to write a design doc or a technical spec.
12. How do you decide when a model is "good enough" to ship?
13. Tell me about a time you had to do a deep dive into data to find a root cause.
14. Describe a feature you engineered that had the biggest impact on a model.
15. Tell me about a time you had to work on a part of the stack you were unfamiliar with (e.g., front-end, data infra).
16. How do you approach testing and validating an ML model before deployment?
17. Tell me about a project that didn't have a clear "right answer."
18. What's the most complex algorithm you've had to implement or debug?

---

### 2. Teamwork & Collaboration

*(How you work with other people)*

19. Tell me about a time you had to work on a project with a cross-functional team (e.g., product managers, designers, data engineers).
20. How did you handle communication with non-technical stakeholders?
21. Describe a time you had to rely on someone else's work to get your job done.
22. Tell me about a time you helped a teammate who was struggling.
23. Describe a situation where you had to work with a difficult teammate.
24. Tell me about a time you had to get buy-in from multiple stakeholders.
25. Describe a time you went out of your way to help your team succeed.
26. Tell me about a time you received difficult feedback from a teammate or manager.
27. How do you build trust with your teammates?
28. Tell me about a time you had to onboard or mentor a new team member.
29. Describe a time you improved a team process (e.g., code reviews, meetings, documentation).
30. Tell me about a time you had to collaborate with a remote team or someone in a different time zone.
31. How do you share your work and knowledge with your team?
32. Tell me about a time you had to take on "glue work" (e.g., documentation, improving CI/CD) for the good of the team.
33. Describe a time you contributed to a project that wasn't your direct responsibility.

---

### 3. Conflict & Disagreement

*(How you handle friction and push back)*

34. Tell me about a time you disagreed with your manager.
35. Describe a time you had a technical disagreement with a peer or a senior engineer.
36. Walk me through a situation where you had to defend a decision you made.
37. Tell me about a time you received negative feedback you disagreed with.
38. Describe a time when you had to tell a teammate "no."
39. Tell me about a time data or an experiment result contradicted your intuition or a stakeholder's belief.
40. How do you handle code reviews where you have a strong disagreement with the reviewer?
41. Tell me about a time you had to compromise on a technical solution.
42. Describe a time you had to push back on a project deadline or a feature request.
43. What do you do when you and a product manager disagree on a feature's priority?
44. Tell me about a time you were in a meeting and strongly disagreed with the direction. What did you do?
45. Describe a time you had to convince a skeptical person to accept your idea.
46. Tell me about a time you had a conflict and were able to resolve it and build a stronger relationship.

---

### 4. Failure & Mistakes

*(How you own your errors and grow from them)*

47. Tell me about a time you made a mistake at work.
48. Describe a project that failed. Why did it fail, and what did you learn?
49. Tell me about a time you introduced a bug into production. How did you handle it?
50. What's the biggest technical mistake you've made?
51. Describe a time you made a wrong assumption about a project.
52. Tell me about a time your model's performance in production was much worse than in training.
53. Walk me through a time you missed a deadline.
54. Tell me about a time you had to deliver bad news to your team or manager.
55. Describe a time you failed to persuade someone to your point of view.
56. What's the most valuable piece of critical feedback you've ever received?
57. Tell me about a time you built something that nobody used.
58. Describe a time your design or technical choice turned out to be the wrong one.
59. Tell me about a time you had to scrap a project and start over.
60. How do you handle it when you don't know the answer to a question?

---

### 5. Adaptability & Ambiguity

*(How you handle chaos and the unknown)*

61. Tell me about a time you had to work on a project with unclear requirements.
62. Describe a time when a project's priorities suddenly changed.
63. Walk me through a situation where you had to make a decision with incomplete data.
64. Tell me about a time you had to learn a new technology or framework very quickly.
65. Describe a project where the scope changed significantly while you were working on it.
66. Tell me about a time you had to solve a problem you had no idea how to approach.
67. Describe a time you had to "wear multiple hats" and take on responsibilities outside your role.
68. How do you handle being given a vague or open-ended task?
69. Tell me about a time you had to adapt your communication style to a specific audience.
70. Describe a time you were working on a project, and the underlying data source changed.
71. Tell me about the most ambiguous problem you've ever been given.
72. How do you keep moving forward when you're "stuck" on a hard problem?

---

### 6. Leadership & Influence

*(How you take ownership and guide others, *with or without* formal authority)*

73. Tell me about a time you took the initiative to start a project.
74. Describe a time you saw a problem and "owned" it without being asked.
75. Tell me about a time you had to influence a senior engineer or leader.
76. Describe a time you mentored a junior engineer. What was your approach?
77. Tell me about a time you had to motivate a team or a teammate.
78. Walk me through a time you had to "manage up" to help your manager or team.
79. Describe a time you led a project or a feature.
80. Tell me about a time you made a decision that was unpopular but was the right thing to do.
81. Describe a time you improved team standards (e.g., code quality, documentation, testing).
82. Tell me about a time you had to build consensus within a group to move forward.
83. How do you show ownership in your day-to-day work?
84. Describe a time you had to delegate a task. How did you ensure it was done correctly?
85. Tell me about a time you represented your team in a cross-functional meeting.

---

### 7. Time Management & Prioritization

*(How you organize your time and effort)*

86. How do you prioritize your work when you have multiple competing tasks?
87. Tell me about a time you had to make a trade-off between quality and speed.
88. Describe a time you had to manage multiple projects at the same time.
89. Walk me through a typical week. How do you structure your time?
90. Tell me about a time you had to work under a very tight deadline.
91. How do you handle distractions?
92. Tell me about a time you had to cut scope to meet a deadline.
93. How do you decide what to work on first in the morning?
94. Tell me about a long-term project. How did you plan it and track your progress?
95. Describe a time you proactively identified a potential bottleneck or risk in a project.
96. How do you balance new feature development with paying down technical debt?
97. Tell me about a time you felt overwhelmed with work. What did you do?

---

### 8. Curiosity & Learning

*(How you grow and stay sharp)*

98. Tell me about a time you went "above and beyond" what was expected of you.
99. Describe a time you challenged the status quo or how things were "always done."
100. Tell me about a new ML paper or technique you've learned about recently.
101. How do you keep your technical skills up to date?
102. Tell me about a time you learned something from a junior teammate.
103. Describe a time you were curious about a part of the business and learned more about it.
104. Tell me about a "pet project" you've worked on in your own time.
105. What is a piece of feedback you've received that led you to learn something new?
106. Tell me about a time you automated a tedious task.
107. Describe a time you dove deep into a technical problem to understand its root cause.
108. What's a new skill (technical or otherwise) you're trying to learn right now?

---

### 9. Communication

*(How you articulate and receive ideas)*

109. Tell me about a time you had to explain a complex technical concept (like backpropagation or transformers) to a non-technical person.
110. Describe a time you had to present your work to a large group.
111. Tell me about a piece of documentation you've written.
112. How do you ensure your code reviews are constructive?
113. Describe a time you had to communicate a project's status, including risks.
114. Tell me about a time you had to ask clarifying questions to understand a requirement.
115. Describe a time you persuaded someone with data.
116. How do you use visualizations to communicate your findings?
117. Tell me about a time a miscommunication caused a problem. How did you fix it?

---

### 10. Culture & Motivation

*(Why you are here and what drives you)*

118. Why do you want to work for [This Company]?
119. What about this specific role interests you?
120. What is your favorite [Company] product, and how would you improve it?
121. Why are you passionate about machine learning?
122. What kind of team environment do you thrive in?
123. What are your career goals for the next 5 years?
124. What's the most interesting problem you've ever worked on?
125. What does "working at scale" mean to you?
126. Tell me about a time you had to work on a task you found boring.
127. What's your ideal project?
128. What's a hard problem you're looking forward to solving?
129. What are you looking for in your next manager?
130. What questions do you have for me? (Always have good questions ready!)
