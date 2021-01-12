# awesome-implicit-neural-models
A collection of resources on *Implicit* learning model, ranging from Neural ODEs to Equilibrium Networks, Differentiable Optimization Layers and more.

*"The crux of an **implicit layer**, is that instead of specifying how to compute the layer’s output from the input, we specify the conditions that we want the layer’s output to satisfy."* [cit.](http://implicit-layers-tutorial.org/)

**NOTE:** Feel free to suggest additions via `Issues` or `Pull Requests`.

***For a comprehensive list of resources on the connections between differential equations and deep learning, please refer to [awesome-neural-ode](https://github.com/Zymrael/awesome-neural-ode)***

# Table of Contents

* **Implicit Deep Learning**

	* [Neural Differential Equations](#neural-differential-Equations)
	
	* [Deep Equilibrium Networks](#deep-equilibrium-networks)
	
	* [Optimization Layers](#optimization-layers)

	
* **Additional Material**
  * [Software and Libraries](#software-and-libraries)
  * [Tutorials and Talks](#tutorials-and-talks)

## Implicit Deep Learning

### Neural Differential Equations
**In *Neural Differential Equations*, the input-output mapping is realized by solving a boundary value problem. The learnable component is the *vector field* of the differential equation.**

* Neural Ordinary Differential Equations (best paper award): [NeurIPS18](https://arxiv.org/pdf/1806.07366.pdf)

> We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions

* Dissecting Neural ODEs (oral): [NeurIPS20](https://arxiv.org/abs/2002.08071)

> Continuous deep learning architectures have recently re-emerged as Neural Ordinary Differential Equations (Neural ODEs). This infinite-depth approach theoretically bridges the gap between deep learning and dynamical systems, offering a novel perspective. However, deciphering the inner working of these models is still an open challenge, as most applications apply them as generic black-box modules. In this work we "open the box", further developing the continuous-depth formulation with the aim of clarifying the influence of several design choices on the underlying dynamics. 

* Neural Controlled Differential Equations for Irregular Time Series (spotlight): [NeurIPS20](https://arxiv.org/abs/2005.08926)

> Neural ordinary differential equations are an attractive option for modelling temporal dynamics. However, a fundamental issue is that the solution to an ordinary differential equation is determined by its initial condition, and there is no mechanism for adjusting the trajectory based on subsequent observations. Here, we demonstrate how this may be resolved through the well-understood mathematics of *controlled differential equations*. The resulting *neural controlled differential equation* model is directly applicable to the general setting of partially-observed irregularly-sampled multivariate time series, and (unlike previous work on this problem) it may utilise memory-efficient adjoint-based backpropagation even across observations. We demonstrate that our model achieves state-of-the-art performance against similar (ODE or RNN based) models in empirical studies on a range of datasets. Finally we provide theoretical results demonstrating universal approximation, and that our model subsumes alternative ODE models. 

* Scalable Gradients for Stochastic Differential Equations: [AISTATS20](https://arxiv.org/abs/2001.01328)

> The adjoint sensitivity method scalably computes gradients of solutions to ordinary differential equations. We generalize this method to stochastic differential equations, allowing time-efficient and constant-memory computation of gradients with high-order adaptive solvers. Specifically, we derive a stochastic differential equation whose solution is the gradient, a memory-efficient algorithm for caching noise, and conditions under which numerical solutions converge. In addition, we combine our method with gradient-based stochastic variational inference for latent stochastic differential equations. We use our method to fit stochastic dynamics defined by neural networks, achieving competitive performance on a 50-dimensional motion capture dataset. 

### Deep Equilibrium Networks
**In *Equilibrium Models*, the output of the model must be a fixed point of some learnable transaformation (e.g. a discrete time map), often explicitly dependent on the input.**

* Deep Equilibrium Models: [NeurIPS19](https://arxiv.org/abs/1909.01377)

> We present a new approach to modeling sequential data: the deep equilibrium model (DEQ). Motivated by an observation that the hidden layers of many existing deep sequence models converge towards some fixed point, we propose the DEQ approach that directly finds these equilibrium points via root-finding.

* Multiscale Deep Equilibrium Models: [NeurIPS20](https://arxiv.org/abs/2006.08656)

* Monotone Operator Equilibrium Networks: [NeurIPS20](https://arxiv.org/abs/2006.08591)

* Lipschitz Bounded Equilibrium Networks: [Arxiv](https://arxiv.org/abs/2010.01732)

* Implicit Deep Learning: [Arxiv](https://arxiv.org/abs/1908.06315)

### Optimization Layers
**To infer any *differentiable optimization layer*, some cost function has to be minimized (maximized) **

* OptNet: Differentiable Optimization as a Layer in Neural Networks: [ICML17](https://arxiv.org/abs/1703.00443)

* Input Covex Neural Networks [ICML17](http://proceedings.mlr.press/v70/amos17b/amos17b.pdf) 

* Differentiable MPC for End-to-end Planning and Control: [NeurIPS18](https://papers.nips.cc/paper/2018/file/ba6d843eb4251a4526ce65d1807a9309-Paper.pdf)

* Differentiable Convex Optimization Layers: [NeurIPS18](https://papers.nips.cc/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html)

* Differentiable Implicit Layers: [NeurIPS20](https://arxiv.org/pdf/2010.07078.pdf)

> In  this  paper,   we  introduce  an  efficient  backpropagation  scheme  for  non-constrained implicit functions. These functions are parametrized by a set of learn-able weights and may optionally depend on some input; making them perfectlysuitable as a learnable layer in a neural network.  We demonstrate our scheme ondifferent applications:  (i) neural ODEs with the implicit Euler method, and (ii) system identification in model predictive control.


## Additional Material

### Software and Libraries
#### Neural ODEs
* `torchdiffeq` Differentiable ODE solvers with full GPU support and `O(1)`-memory backpropagation: [repo](https://github.com/rtqichen/torchdiffeq)
* `torchdyn` PyTorch library for all things neural differential equations. [repo](https://github.com/diffeqml/torchdyn), [docs](https://torchdyn.readthedocs.io/)
* `torchsde` Stochastic differential equation (SDE) solvers with GPU support and efficient sensitivity analysis: [repo](https://github.com/google-research/torchsde)
* `torchcde` GPU-capable solvers for controlled differential equations (CDEs): [repo](https://github.com/patrick-kidger/torchcde)


#### Deep Equilibrium Models
* `deq` This repository contains the code for the deep equilibrium (DEQ) model, an implicit-depth architecture [repo](https://github.com/locuslab/deq)

* `deq-jax` Jax Implementation for the deep equilibrium (DEQ) model [repo](https://github.com/akbir/deq-jax)

#### Optimization

* `mpc.pytorch` A fast and differentiable model predictive control solver for PyTorch. [repo](https://github.com/locuslab/mpc.pytorch), [docs](https://locuslab.github.io/mpc.pytorch/)

## Tutorials and Talks

* **NeurIPS20 Tutorial** *Deep Implicit Layers - Neural ODEs, Deep Equilibirum Models, and Beyond* [website](http://implicit-layers-tutorial.org/)
