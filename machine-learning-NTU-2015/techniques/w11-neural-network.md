# Neural Network

### Notation
* 3-5-1 NNM -> bias term is no included.
 * So the # of weights is $$(3+1) \times 5 + (5+1) \times 1$$


### Backpropagation(Backprop) Algorithm
#### **TL;DR**:
**除了output x (input of error function), 一組x 配一組w。 s 和 $$\sigma$$ 1-to-1 mapping**


* Note that index definition: $$x^{(0)}$$ is raw input data

* Predictions

> $$(x^{(0)T}w^{(1)})^T= w^{(1)T}x^{(0)}=s^{(1)}$$

> $$w^{(l)T}x^{(l-1)}=s^{(l)}$$

> $$\tanh(s^{(1)}) = x^{(1)}$$

* Train (learn $$w^{(l)} ,\: 1\leq l\leq L$$)

