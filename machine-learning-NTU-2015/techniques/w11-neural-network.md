# Neural Network

### Notation
* 3-5-1 NNM -> bias term is no included.
 * So the # of weights is $$(3+1) \times 5 + (5+1) \times 1$$


### Backpropagation(Backprop) Algorithm
#### **TL;DR**:
**除了output x (input of error function), 一組x 配一組w。 s 和 $$\delta$$ 1-to-1 mapping**


* Note that index definition: $$x^{(0)}$$ is raw input data. And the input data is a **row vector**

* Predictions

> $$x^{(0)}w^{(1)}=s^{(1)}$$

> $$x^{(l-1)}w^{(l)}=s^{(l)}=\phi^{(l)}(x)$$

> $$\tanh(s^{(1)}) = x^{(1)}$$

> $$\tanh(s^{(l)}) = x^{(l)}$$

> $$ x^{l}*w^{l+1} = s^{l+1} $$

> $$\frac{\partial e }{\partial s^l_j} = \delta^l_j$$

For $$L$$ as the # of layers of neurons (e.g. 1-4-1 NNM -> L=3)

> $$s^l, w^l, \delta^l_j, 1 \leq l < L$$

> $$x^l, 0\leq l < L$$

> Train Target (learn $$w^{(l)} ,\: 1\leq l < L$$)

#### <a name="Pesudo"></a> Pseudo Code:

> Input: $$D$$ as training data, $$\eta$$ as learning step

> Integer array $$N$$, indicating the number of neurons in each layer. Last layer must be **one** and first layers should match the dimension of Training data

> Initialize $$W$$ by random number

> Randomly pick up $$x \in D$$

> Update $$S=[s^1,...,s^{L-1}]$$ and its non-linear transformed $$X=[x^1,...,x^{L-1}]$$

> Update $$\Delta = [\delta^1,..., \delta^{L-1}]$$ By

>> $$\delta_j^{L-1} = \frac{\partial e}{\partial s_j^{l-1}}$$ directly

>> For $$ 1 \leq l < L-1$$, $$\delta^l = s^l * w^l * tanh'(s^l) $$

> Update W by

>> for each $$w^l$$

>> $$w^l = w^l - \eta * (x^{l-1})^T * \delta^l$$

