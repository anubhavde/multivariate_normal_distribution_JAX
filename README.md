##### **congenial-computing-machine**
# **Multivariate Normal Distribution JAX**

Implemented from scratch a sampling method to draw samples from a multivariate Normal (MVN) distribution in JAX.

Rules for this project were:-
- Code should work for any number of dimensions but please set the number of dimensions (random variables of MVN) to 10 for this task.

- Only allowed to use ```jax.random.uniform```. Especially not allowed to use ```jax.random.normal```.

- Should randomly create the mean and covariance matrix to fully specify an MVN distribution.

- Implement a sampling method from scratch using which you can draw samples from the specified MVN distribution.

- Use sampling method to draw multiple samples from the MVN distribution and reconstruct the parameters of your MVN distribution (mean and covariance matrix) to confirm that your sampling method is working correctly.
