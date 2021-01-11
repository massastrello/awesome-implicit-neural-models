# awesome-implicit-neural-models
A collection of resources on *Implicit* learning model, ranging from Neural ODEs to Equilibrium Networks, Differentiable Optimization Layers and more.

**NOTE:** Feel free to suggest additions via `Issues` or `Pull Requests`.

# Table of Contents

* **Implicit Deep Learning**

	* [Neural Differential Equations](#neural-differential-Equations)
	
	* [Deep Equilibrium Networks](#deep-equilibrium-networks)
	
	* [Optimization Layers](#optimization-layers)

	
* **Additional Material**
  * [Software and Libraries](#software-and-libraries)

  * [Websites and Blogs](#websites-and-blogs)

## Implicit Deep Learning

### Neural Differential Equations

* Neural Ordinary Differential Equations (best paper award): [NeurIPS18](https://arxiv.org/pdf/1806.07366.pdf)

> We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions

* Dissecting Neural ODEs (oral): [NeurIPS20](https://arxiv.org/abs/2002.08071)

> In this work, we “open the box” and offer a system–theoretic perspective with the aim of clarifying the influence of several design choices on the underlying dynamics. We formulate and solve the infinite–dimensional problem linked to the true deep limit formulation of Neural ODE. We provide numerical approximations to the infinite–dimensional problem, leading to novel model variants, such as Galerkin and piecewise–constant Neural ODEs. Augmentation is developed beyond existing approaches to include input–layer and higher–order augmentation strategies. Finally, the novel paradigms of data–control (vector field conditioning) and depth–adaptation are introduced.

* Neural Controlled Differential Equations for Irregular Time Series (spotlight): [NeurIPS20](https://arxiv.org/abs/2005.08926)

> Neural ordinary differential equations are an attractive option for modelling temporal dynamics. However, a fundamental issue is that the solution to an ordinary differential equation is determined by its initial condition, and there is no mechanism for adjusting the trajectory based on subsequent observations. Here, we demonstrate how this may be resolved through the well-understood mathematics of *controlled differential equations*. The resulting *neural controlled differential equation* model is directly applicable to the general setting of partially-observed irregularly-sampled multivariate time series, and (unlike previous work on this problem) it may utilise memory-efficient adjoint-based backpropagation even across observations. We demonstrate that our model achieves state-of-the-art performance against similar (ODE or RNN based) models in empirical studies on a range of datasets. Finally we provide theoretical results demonstrating universal approximation, and that our model subsumes alternative ODE models. 

Scalable Gradients for Stochastic Differential Equations: [AISTATS20](https://arxiv.org/abs/2001.01328)

> The adjoint sensitivity method scalably computes gradients of solutions to ordinary differential equations. We generalize this method to stochastic differential equations, allowing time-efficient and constant-memory computation of gradients with high-order adaptive solvers. Specifically, we derive a stochastic differential equation whose solution is the gradient, a memory-efficient algorithm for caching noise, and conditions under which numerical solutions converge. In addition, we combine our method with gradient-based stochastic variational inference for latent stochastic differential equations. We use our method to fit stochastic dynamics defined by neural networks, achieving competitive performance on a 50-dimensional motion capture dataset. 

`For a comprehensive list of resources on the connections between differential equations and deep learning, please refer to `

### Deep Equilibrium Networks

* Deep Equilibrium Models: [NeurIPS19](https://arxiv.org/abs/1909.01377)

> We present a new approach to modeling sequential data: the deep equilibrium model (DEQ). Motivated by an observation that the hidden layers of many existing deep sequence models converge towards some fixed point, we propose the DEQ approach that directly finds these equilibrium points via root-finding.

### Optimization Layers



## Additional Material

### Software and Libraries

### Websites and Blogs
