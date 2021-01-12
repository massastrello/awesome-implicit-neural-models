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

* Neural Controlled Differential Equations for Irregular Time Series (spotlight): [NeurIPS20](https://arxiv.org/abs/2005.08926)

> Neural ordinary differential equations are an attractive option for modelling temporal dynamics. However, a fundamental issue is that the solution to an ordinary differential equation is determined by its initial condition, and there is no mechanism for adjusting the trajectory based on subsequent observations. Here, we demonstrate how this may be resolved through the well-understood mathematics of *controlled differential equations*. The resulting *neural controlled differential equation* model is directly applicable to the general setting of partially-observed irregularly-sampled multivariate time series, and (unlike previous work on this problem) it may utilise memory-efficient adjoint-based backpropagation even across observations. We demonstrate that our model achieves state-of-the-art performance against similar (ODE or RNN based) models in empirical studies on a range of datasets. Finally we provide theoretical results demonstrating universal approximation, and that our model subsumes alternative ODE models. 

Scalable Gradients for Stochastic Differential Equations: [AISTATS20](https://arxiv.org/abs/2001.01328)

> The adjoint sensitivity method scalably computes gradients of solutions to ordinary differential equations. We generalize this method to stochastic differential equations, allowing time-efficient and constant-memory computation of gradients with high-order adaptive solvers. Specifically, we derive a stochastic differential equation whose solution is the gradient, a memory-efficient algorithm for caching noise, and conditions under which numerical solutions converge. In addition, we combine our method with gradient-based stochastic variational inference for latent stochastic differential equations. We use our method to fit stochastic dynamics defined by neural networks, achieving competitive performance on a 50-dimensional motion capture dataset. 

`For a comprehensive list of resources on the connections between differential equations and deep learning, please refer to `

### Deep Equilibrium Networks

* Deep Equilibrium Models: [NeurIPS19](https://arxiv.org/abs/1909.01377)

> We present a new approach to modeling sequential data: the deep equilibrium model (DEQ). Motivated by an observation that the hidden layers of many existing deep sequence models converge towards some fixed point, we propose the DEQ approach that directly finds these equilibrium points via root-finding.

* Multiscale Deep Equilibrium Models: [NeurIPS20](https://arxiv.org/abs/2006.08656)

* Monotone Operator Equilibrium Networks: [NeurIPS20](https://arxiv.org/abs/2006.08591)

* Lipschitz Bounded Equilibrium Networks: [Arxiv](https://arxiv.org/abs/2010.01732)

### Optimization Layers

* OptNet: Differentiable Optimization as a Layer in Neural Networks: [ICML17](https://arxiv.org/abs/1703.00443)

* Differentiable Convex Optimization Layers: [NeurIPS18](https://papers.nips.cc/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html)

* Differentiable Implicit Layers: [NeurIPS20](https://arxiv.org/pdf/2010.07078.pdf)

> In  this  paper,   we  introduce  an  efficient  backpropagation  scheme  for  non-constrained implicit functions. These functions are parametrized by a set of learn-able weights and may optionally depend on some input; making them perfectlysuitable as a learnable layer in a neural network.  We demonstrate our scheme ondifferent applications:  (i) neural ODEs with the implicit Euler method, and (ii) system identification in model predictive control.


## Additional Material

### Software and Libraries

### Websites and Blogs
