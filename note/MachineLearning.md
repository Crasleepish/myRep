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
> $$
\begin{align*} \text{repeat until convergence: } \lbrace & \newline \theta_0 := & \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) \newline \theta_1 := & \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i}) x_{i}\right) \newline \rbrace& \end{align*}
> $$
> This method looks at every example in the entire training set on every step, and is called **batch gradient descent**.




# Multivariate Linear Regression

## Multiple Features

Linear regression with multiple variables is also known as "multivariate linear regression".
$$
\begin{align*}x_j^{(i)} &= \text{value of feature } j \text{ in the }i^{th}\text{ training example} \newline x^{(i)}& = \text{the input (features) of the }i^{th}\text{ training example} \newline m &= \text{the number of training examples} \newline n &= \text{the number of features} \end{align*}
$$
The multivariable form of the hypothesis function accommodating these multiple features is as follows:
$$
h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n
$$
let $x_{0}^{(i)} =1 \text{ for } (i\in { 1,\dots, m } )$, we can represent our pypothesis function as:
$$
\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align*}
$$

## Gradient Descent for Multiple Variables

cost function is as same as previous one variable example:
$$
J(\theta)=J(\theta_0,\theta_1,...,\theta_n)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
$$
where θ is a n-dimension vector.

gradient descent algorithm becomes:
$$
\begin{align*} & \text{repeat until convergence:} \; \lbrace \newline \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline \; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline \; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline & \cdots \newline \rbrace \end{align*}
$$
In other words:
$$
\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\newline \rbrace\end{align*}
$$

> The gradient descent equation can be vectorized as:
> $$
> \theta:=\theta-\frac{\alpha}{m}X^T(X\theta-y)
> $$
> where 
> $$
> X=(x^{(1)},x^{(2)},...,x^{(m)})^T,\theta=(\theta_0,\theta_1,...,\theta_n)^T,y=(y_0,y_1,...y_m)^T
> $$



## Optimization

- Feature Scaling

  We can speed up gradient descent by having each of our input values in roughly the same range. 

  modify the ranges of our input variables so that they are all roughly the same. Ideally: −1 ≤ x~i~ ≤ 1

- Mean normalization

  Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. 

- To implement both of these techniques, adjust your input values as shown in this formula:
  $$
  x_i:=\frac{x_i-\mu_i}{s_i}
  $$
  Where μ~i~ is the **average** of all the values for feature (i) and s~i~ is the range of values (max - min), or s~i~ is the standard deviation.
  
  > According to the property of standard deviation, let σ be standard deviation, and μ be mean, most of entries of data set are distributed between -2σ + μ and 2σ + μ.



## Learning Rate

- **Debugging gradient descent.** Make a plot with *number of iterations* on the x-axis. Now plot the cost function, J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease α.

To summarize:

If α is too small: slow convergence.

If α is too large: ￼may not decrease on every iteration and thus may not converge.



## Features and Polynomial Regression

We can **combine** multiple features into one. For example, we can combine x~1~ and x~2~ into a new feature x~3~ by taking x~1~⋅x~2~

- polynomial regression

  We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

  For example, if our hypothesis function is $h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}$,then we can create additional features based on x~1~: 
  $$
  \begin{aligned}
  h_{\theta}(x)&=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{1}^{2}+\theta_{3} x_{1}^{3}\\
  &=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3,\text { where } x_{2}=x_{1}^{2} \text { and } x_{3}=x_{1}^{3}
  \end{aligned}
  $$
  if you choose your features this way then feature scaling becomes very important.



# Computing Parameters Analytically

## Normal Equation

A second way of minimizing J is to solve θ analytically.
$$
\frac{\partial{J(\theta)}}{\partial{\theta}}=0
$$
make:

$$
X=\begin{pmatrix}x^{(1)T} \\ x^{(2)T} \\ \vdots \\ x^{(m)T} \end{pmatrix},y=(y_0,y_1,...y_m)^T
$$
The normal equation formula is given below:
$$
\theta=\left(X^{T} X\right)^{-1} X^{T} y
$$
This allows us to find the optimum theta without iteration.

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent           | Normal Equation                              |
| :------------------------- | :------------------------------------------- |
| Need to choose alpha       | No need to choose alpha                      |
| Needs many iterations      | No need to iterate                           |
| O (kn^2^)                  | O (n^3^), need to calculate inverse of X^T^X |
| Works well when n is large | Slow if n is very large                      |

If X^T^X is **noninvertible,** the common causes might be having :

- Redundant features, where two features are very closely related (i.e. they are linearly dependent)
- Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).



# Classification and Representation

To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

if we are trying to build a spam classifier for email, then x^{(i)}*x*(*i*) may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, y∈{0,1}. 0 is also called the **negative class**, and 1 the **positive class**, and they are sometimes also denoted by the symbols “-” and “+.” Given x^{(i)}*x*(*i*), the corresponding y^{(i)}*y*(*i*) is also called the **label** for the training example.



## Hypothesis Representation

We use "Sigmoid Function", also called "Logistic Function", to define hypothesis formula:
$$
\begin{align*}& h_\theta (x) = g ( \theta^T x ) \newline \newline& z = \theta^T x \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}
$$
which satisfy $0 \leq h_{\theta}(x) \leq 1$

h~θ~(x) will give us the **probability** that our output is 1. For example, h~θ~(x)=0.7 gives us a probability of 70% that our output is 1.
$$
\begin{align*}& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \newline& P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1\end{align*}
$$

- Decision Boundary

  In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:
  $$
  \begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline& h_\theta(x) < 0.5 \rightarrow y = 0 \newline\end{align*}
  $$
  So if our input to g is θ^T^X, then that means:
  $$
  \begin{align*}& h_\theta(x) = g(\theta^T x) \geq 0.5 \newline& when \; \theta^T x \geq 0\end{align*}
  $$
  we can now say:
  $$
  \begin{align*}& \theta^T x \geq 0 \Rightarrow y = 1 \newline& \theta^T x < 0 \Rightarrow y = 0 \newline\end{align*}
  $$
  The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.



## Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:
$$
\begin{equation}
J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\
\mathrm{Cost}(h_\theta(x),y)=\left\{ \begin{array}{ll}
    -\log(h_\theta(x)) \; & \text{if y = 1} \\
    -\log(1-h_\theta(x)) \; & \text{if y = 0}
    \end{array}
\right.
\end{equation}
$$
Writing the cost function in this way guarantees that J(θ) is convex for logistic regression.

We can compress our cost function's two conditional cases into one case:
$$
\operatorname{cost}\left(h_{\theta}(x), y\right)=-y \log \left(h_{\theta}(x)\right)-(1-y) \log \left(1-h_{\theta}(x)\right)
$$
We can fully write out our entire cost function as follows:
$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
$$
A vectorized implementation is:
$$
\begin{array}{l}{h=g(X \theta)} \\ {J(\theta)=\frac{1}{m} \cdot\left(-y^{T} \log (h)-(1-y)^{T} \log (1-h)\right)}\end{array}
$$



## Gradient Descent

The general form of gradient descent is:
$$
\begin{align*}& Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \newline & \rbrace\end{align*}
$$
We can work out the derivative part using calculus to get:
$$
\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}
$$
Notice that this algorithm is identical to the one we used in linear regression.

A vectorized implementation is:
$$
\theta :=\theta-\frac{\alpha}{m} X^{T}(g(X \theta)-\vec{y})
$$
