# Introduction

- two broad classifications:
  - Supervised learning and Unsupervised learning



## Supervised Learning

- definition:

  data set called "right answers" are given

- category:

  - regression problem（回归问题）

    map input variables to some continuous function

  - classification problem

    map input variables into discrete categories



## Unsupervised Learning

​    unlabled data set

​    We can derive this structure by clustering the data based on relationships among the variables in the data

- examples:
  - Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.
  - Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect)).



# Model and Cost Function

- Univariate linear regression

  Linear regression with one variable

## Cost Function

We can measure the accuracy of our hypothesis function by using a **cost function**.
$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
$$
This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $\left(\frac{1}{2}\right)$as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the$ \frac{1}{2}$ term.

hypothesis:
$$
h_\theta(x)=\theta_0+\theta_1x
$$
parameters:
$$
\theta_0, \theta_1
$$
Cost Function:
$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
$$
Goal:
$$
\min_{\theta_0,\theta_1}J(\theta_0,\theta_1)
$$



## Gradient Descent

Find the local minimum point of the cost function

1. 选定一个初始点$(\theta_0^{(0)},\theta_1^{(0)})$

2. 沿$\nabla{J(\theta_0,\theta_1)}$的反方向移动一段距离（受$\alpha$，称为学习速率，的控制）到$(\theta_0^{(1)},\theta_1^{(1)})$，重复直到收敛

The gradient descent algorithm is:

repeat until convergence:
$$
\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_0}J(\theta_0,\theta_1)
$$
where j=0,1 represents the feature index number.

Regardless of the slope's sign for $\frac{d}{d\theta_1}$ , $\theta_1$eventually converges to its minimum value. 

As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to decrease α over time.



> In our univariate linear regression
$$
\begin{align*} \text{repeat until convergence: } \lbrace & \newline \theta_0 := & \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) \newline \theta_1 := & \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i}) x_{i}\right) \newline \rbrace& \end{align*}
$$
> This method looks at every example in the entire training set on every step, and is called **batch gradient descent**.



