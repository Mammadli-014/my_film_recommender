# 🎬 Custom Film Recommender System

> This README was prepared with the assistance of AI and reviewed and edited by the project author.

---

## 📌 About The Project

The main purpose of this project was to implement a **Collaborative Filtering recommendation system from scratch** using **NumPy**. 

Rather than building a state-of-the-art model, the goal was to understand the inner workings of recommendation systems, focusing on the mathematical foundations behind:
* Matrix Factorization
* Gradient Descent & Momentum
* Biases & Regularization

After building the custom model from scratch, its performance was benchmarked against the `SVD` model from the popular `Surprise` library using the exact same train/test split.

---

## 📊 Dataset

This project utilizes the benchmark **MovieLens 100K** dataset:
* **Users:** 943
* **Movies:** 1,682
* **Total Ratings:** 100,000
* **Rating Scale:** 1 to 5

> **Note:** Although the dataset contains timestamps and other metadata, this implementation strictly relies on `user_id`, `movie_id`, and `rating`. The data was split into **80% training** and **20% testing** sets.

---

## 📐 Model Architecture

The recommendation engine is built on **Matrix Factorization**. Each user and movie is mapped to a latent feature vector:

$$W \in \mathbb{R}^{\text{users} \times \text{features}}$$

$$X \in \mathbb{R}^{\text{movies} \times \text{features}}$$

The predicted rating ($\hat{r}_{ui}$) is calculated as:

$$\hat{r}_{ui} = \mu + b_u + b_i + W_u X_i^T$$

**Where:**
* $\mu$ = Global mean rating
* $b_u$ = User bias
* $b_i$ = Movie bias
* $W_u$ = User latent feature vector
* $X_i$ = Movie latent feature vector

---

## 📉 Loss Function & Regularization

The model minimizes the squared error exclusively for the ratings that actually exist in the dataset:

$$J_{\text{data}} = \frac{1}{2} \sum_{u,i} R_{ui} (\hat{r}_{ui} - r_{ui})^2$$

*(where $R$ is a mask indicating the presence of a rating).*

### L2 Regularization
To prevent overfitting, L2 regularization is applied to all parameters:

$$J_{\text{reg}} = \frac{\lambda}{2} \left( \Vert{}W\Vert{}^2 + \Vert{}X\Vert{}^2 + \Vert{}b_u\Vert{}^2 + \Vert{}b_i\Vert{}^2 \right)$$

Combining both gives the final objective function:

$$J = J_{\text{data}} + J_{\text{reg}}$$

---

## ⚙️ Optimization (Gradient Descent & Momentum)

Parameters are optimized using **Gradient Descent** coupled with **Momentum** for smoother convergence and stability.

### Gradients
$$\frac{\partial J}{\partial W} = EX + \lambda W$$

$$\frac{\partial J}{\partial X} = E^T W + \lambda X$$

$$\frac{\partial J}{\partial b_u} = \sum_i E_{ui} + \lambda b_u$$

$$\frac{\partial J}{\partial b_i} = \sum_u E_{ui} + \lambda b_i$$

### Momentum Update
The velocity update follows:

$$v_t = \beta v_{t-1} + \alpha \nabla J$$

$$\theta_t = \theta_{t-1} - v_t$$

> *Implementation Detail:* A momentum value ($\beta$) of **0.8** was used during training.

---

## 📈 Evaluation Metrics

The model performance is evaluated using:
* **Root Mean Squared Error (RMSE)**
* **Mean Squared Error (MSE)**

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{u,i} (\hat{r}_{ui} - r_{ui})^2}$$

$$\text{MSE} = \frac{1}{N} \sum (\hat{r} - r)^2$$

---

## 🏆 Results & Comparison

Below is the performance comparison between the custom NumPy recommender and the `Surprise` library's SVD model under identical evaluation splits.

### 1. Custom Film Recommender (Scratch)
```text
RMSE: 0.9345
MSE:  0.8732
```

### SVD model
```
RMSE: 0.9327
MAE:  0.7349