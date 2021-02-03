# awesome-implicit-neural-models
A collection of resources on *Implicit* learning model, ranging from Neural ODEs to Equilibrium Networks, Differentiable Optimization Layers and more.

*"The crux of an **implicit layer**, is that instead of specifying how to compute the layer’s output from the input, we specify the conditions that we want the layer’s output to satisfy."* [cit. (NeurIPS 2020 Implicit Layers Tutorial)](http://implicit-layers-tutorial.org/)

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

* Discretize-Optimize vs. Optimize-Discretize for Time-Series Regression and Continuous Normalizing Flows: [arxiv](https://arxiv.org/abs/2005.13420)

> We compare the discretize-optimize (Disc-Opt) and optimize-discretize (Opt-Disc) approaches for time-series regression and continuous normalizing flows (CNFs) using neural ODEs. Neural ODEs are ordinary differential equations (ODEs) with neural network components. Training a neural ODE is an optimal control problem where the weights are the controls and the hidden features are the states. Every training iteration involves solving an ODE forward and another backward in time, which can require large amounts of computation, time, and memory. Comparing the Opt-Disc and Disc-Opt approaches in image classification tasks, Gholami et al. (2019) suggest that Disc-Opt is preferable due to the guaranteed accuracy of gradients. In this paper, we extend the comparison to neural ODEs for time-series regression and CNFs. Unlike in classification, meaningful models in these tasks must also satisfy additional requirements beyond accurate final-time output, e.g., the invertibility of the CNF. Through our numerical experiments, we demonstrate that with careful numerical treatment, Disc-Opt methods can achieve similar performance as Opt-Disc at inference with drastically reduced training costs. Disc-Opt reduced costs in six out of seven separate problems with training time reduction ranging from 39% to 97%, and in one case, Disc-Opt reduced training from nine days to less than one day.

* Bayesian Neural ODEs: [POPL2021-LAFI](https://arxiv.org/pdf/2012.07244.pdf)

> Recently, Neural Ordinary Differential Equations has emerged as a powerful framework for modeling physical simulations without explicitly defining the ODEs governing the system, but learning them via machine learning. However, the question: “Can Bayesian learning frameworks be integrated with Neural ODE’s to robustly quantify the uncertainty in the weights of a Neural ODE?” remains unanswered. In an effort to address this question, we demonstrate the successful integration of Neural ODEs with two methods of Bayesian Inference: (a) The No-U-Turn MCMC sampler (NUTS) and (b) Stochastic Langevin Gradient Descent (SGLD). We test the performance of our Bayesian Neural ODE approach on classical physical systems, as well as on standard machine learning datasets like MNIST, using GPU acceleration. Finally, considering a simple example, we demonstrate the probabilistic identification of model specification in partially-described dynamical systems using universal ordinary differential equations. Together, this gives a scientific machine learning tool for probabilistic estimation of epistemic uncertainties.

### Deep Equilibrium Networks
**In *Equilibrium Models*, the output of the model must be a fixed point of some learnable transaformation (e.g. a discrete time map), often explicitly dependent on the input.**

* Deep Equilibrium Models: [NeurIPS19](https://arxiv.org/abs/1909.01377)

> We present a new approach to modeling sequential data: the deep equilibrium model (DEQ). Motivated by an observation that the hidden layers of many existing deep sequence models converge towards some fixed point, we propose the DEQ approach that directly finds these equilibrium points via root-finding.

* Multiscale Deep Equilibrium Models: [NeurIPS20](https://arxiv.org/abs/2006.08656)

* Monotone Operator Equilibrium Networks: [NeurIPS20](https://arxiv.org/abs/2006.08591)

* Lipschitz Bounded Equilibrium Networks: [Arxiv](https://arxiv.org/abs/2010.01732)

* Implicit Deep Learning: [Arxiv](https://arxiv.org/abs/1908.06315)

* Algorithmic Differentiation of a Complex C++ Code with Underlying Libraries (An AD system for C++ with DEQ-like adjoints by default on PETSc) [Paper](https://www.sciencedirect.com/science/article/pii/S187705091300327X)

### Optimization Layers
**To infer any *differentiable optimization layer*, some cost function has to be minimized (maximized)**

* OptNet: Differentiable Optimization as a Layer in Neural Networks: [ICML17](https://arxiv.org/abs/1703.00443)

* Input Covex Neural Networks [ICML17](http://proceedings.mlr.press/v70/amos17b/amos17b.pdf) 

* Differentiable MPC for End-to-end Planning and Control: [NeurIPS18](https://papers.nips.cc/paper/2018/file/ba6d843eb4251a4526ce65d1807a9309-Paper.pdf)

* Differentiable Convex Optimization Layers: [NeurIPS19](https://papers.nips.cc/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html)

* Differentiable Implicit Layers: [NeurIPS20](https://arxiv.org/pdf/2010.07078.pdf)

> In  this  paper,   we  introduce  an  efficient  backpropagation  scheme  for  non-constrained implicit functions. These functions are parametrized by a set of learn-able weights and may optionally depend on some input; making them perfectlysuitable as a learnable layer in a neural network.  We demonstrate our scheme ondifferent applications:  (i) neural ODEs with the implicit Euler method, and (ii) system identification in model predictive control.


## Additional Material

### Software and Libraries
#### Neural ODEs
* `torchdiffeq` Differentiable ODE solvers with full GPU support and `O(1)`-memory backpropagation: [repo](https://github.com/rtqichen/torchdiffeq)
* `torchdyn` PyTorch library for all things neural differential equations. [repo](https://github.com/diffeqml/torchdyn), [docs](https://torchdyn.readthedocs.io/)
* `torchsde` Stochastic differential equation (SDE) solvers with GPU support and efficient sensitivity analysis: [repo](https://github.com/google-research/torchsde)
* `torchcde` GPU-capable solvers for controlled differential equations (CDEs): [repo](https://github.com/patrick-kidger/torchcde)
* `DifferentialEquations.jl` is a set of ODE/SDE/DAE/DDE/jump/etc. solvers with GPU and distributed computing support, event handling, along with O(1) memory adjoints and stabilized versions for stiff and partial differential equations [repo](https://github.com/SciML/DifferentialEquations.jl), [docs](https://diffeq.sciml.ai/stable/)
* `DiffEqFlux.jl` is a companion library to `DifferentialEquations.jl` which includes common implicit layer models and tooling such as collocation schemes for building complex loss functions [repo](https://github.com/SciML/DiffEqFlux.jl), [docs](https://diffeqflux.sciml.ai/dev/)

#### Deep Equilibrium Models
* `deq` This repository contains the code for the deep equilibrium (DEQ) model, an implicit-depth architecture [repo](https://github.com/locuslab/deq)
* `deq-jax` Jax Implementation for the deep equilibrium (DEQ) model [repo](https://github.com/akbir/deq-jax)
* `DifferentialEquations.jl` `SteadyStateProblem` is a differentiable solver for steady-states of differential equations [repo](https://github.com/SciML/DifferentialEquations.jl)

#### Optimization

* `mpc.pytorch` A fast and differentiable model predictive control solver for PyTorch. [repo](https://github.com/locuslab/mpc.pytorch), [docs](https://locuslab.github.io/mpc.pytorch/)
* `cvxpylayers` Differentiable convex optimization layers in PyTorch and TensorFlow using CVXPY. [repo](https://github.com/cvxgrp/cvxpylayers)

## Tutorials and Talks

* **NeurIPS20 Tutorial** *Deep Implicit Layers - Neural ODEs, Deep Equilibirum Models, and Beyond* [website](http://implicit-layers-tutorial.org/)
* **JuliaCon 2019** *Neural Ordinary Differential Equations with DiffEqFlux | Jesse Bettencourt* [youtube](https://youtu.be/5ZgEp36E71Y)
