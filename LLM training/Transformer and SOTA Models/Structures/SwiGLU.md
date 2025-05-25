
## 1. Introduction

Google的[PaLM](https://zhida.zhihu.com/search?content_id=232627022&content_type=Article&match_order=1&q=PaLM&zhida_source=entity)和Meta的[LLaMA](https://zhida.zhihu.com/search?content_id=232627022&content_type=Article&match_order=1&q=LLaMA&zhida_source=entity)等模型使用SwigLU替换了FFN模块，可以很好利用门控网络的优点选择性过滤信息，在众多LLM Model中被广泛使用。

> [GLU 和 SwiGLU](https://mingchao.wang/1fb1JNJ6/)
> [大模型基础｜激活函数｜从ReLU 到SwiGLU](https://zhuanlan.zhihu.com/p/650237644)
## 2. Relu/GELU/Swish 激活函数

ReLU是传统神经网络最常见的激活函数，实现简单速度快，可以缓解梯度消失的问题，但可能带来负输入的梯度为0，导致神经元无法更新的问题。[GELU](https://arxiv.org/abs/1606.08415) 结合输入的概率权重，对负值有自适应调整，在自然语言处理（如BERT、GPT）中表现优异。Swish 负区间内微小可导，相对于relu更加平滑，也可缓解relu负值无激活的问题。相关计算公式和函数如下：
$$

\begin{aligned}

ReLU(x) &= max(x, 0) \\ 
Swish(x) &= x\cdot \sigma(\beta x) \\ 
GLUE(x) &= x P(X \leq x) = x \cdot \frac{1}{2} [1 + erf(x/\sqrt{2})]

\end{aligned}
$$

![[activation_functions.png]]

## 3. GLU

> ref [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083.pdf)

### 3.1 GLU key idea

- 通过门控网络过滤或放大输入 $x$ 各个维度信息。
- 可以理解成利用 $x$ 作为输入通过一个简单的线性层训练一个输入向量级别的 $elementwise- attention$。
### 3.2 原始GLU公式

从向量维度理解，假设输入向量 $x(1, n)$：
- 通过两个线性变换得到：$\alpha=xW+b$，$\beta=xV+c$。
- 计算 $GLU(x)$
$$
GLU(x)=\alpha \otimes \sigma(\beta)=(x⋅W+b) \otimes \sigma(x⋅V+c)
$$
- $\sigma(x\cdot V + c)$ 为数值范围在 $[0, 1]$ ，形状为$(1,n)$的向量，用来控制 $\alpha$ 的每个维度输出到下一层的rate。

![[GLU.png]]
### 3.3 GLU激活函数

GLU作为激活函数时不将输入作线性变换，只训练 $\sigma(x\cdot V + c)$ 部分参数，公式如下：
$$
GLU(x) = x \ \otimes \sigma(x⋅V+c)
$$

## 4. SwiGLU

### 4.1 传统FFN公式及变体

一般是有两个MLP层，先升维再降维，以ReLU作为激活函数为例：
$$
FFN(x,W1,W2,b1,b2)=ReLU(xW1+b1)W2+b2
$$
有些研究，比如T5模型，去掉了bias，简化为：
$$
FFN_{ReLU}(x,W1,W2)=ReLU(xW1)W2
$$
同样不考虑bias，可以把以上case拓展到 $GeLU$ 和 $Swish$ 的场景。
$$
\begin{aligned}
FFN_{GeLU}(x,W1,W2)=GeLU(xW1)W2 \\

FFN_{Swish}(x,W1,W2)=Swish(xW1)W2
\end{aligned}
$$

### 4.2 GLU及变体

传统GLU公式为：
$$
GLU(x,W,V,b,c)=σ(xW+b)⊗(xV+c)
$$
等式右边 $\otimes$ 左侧看作gating部分，用于门控，这部分使用不同的函数可以衍生出不同GLU的变体：
$$
\begin{aligned}
Bilinear(x,W,V,b,c)&=(xW+b)\otimes(xV+c) \\

ReGLU(x,W,V,b,c)&=ReLU(xW+b)\otimes(xV+c) \\

GEGLU(x,W,V,b,c)&=GELU(xW+b)\otimes(xV+c) \\

SwiGLU(x,W,V,b,c,β)&=Swishβ(xW+b)\otimes(xV+c) \\
\end{aligned}
$$

### 4.3 FFN 模块的表示

不考虑bias偏置的情况下，整个FFN模块通常在升维时使用 $SwiGLU$ 激活函数。

$$
\begin{aligned}
FFN_{GLU}(x,W,V)&=[\sigma(xW)\otimes(xV)]W2 \\
FFN_{Bilinear}(x,W,V)&=[(xW)\otimes(xV)]W2 \\
FFN_{ReGLU}(x,W,V)&=[ReLU(xW)\otimes(xV)]W2 \\
FFN_{GEGLU}(x,W,V)&=[GELU(xW)\otimes(xV)]W2 \\
FFN_{SwiGLU}(x,W,V)&=[Swish(xW)\otimes(xV)]W2 \\
\end{aligned}
$$

### 4.4 使用SwiGLU之后的维度变化

假设FFN模块的输入$[b, s]$，中间维度为$d$，使用ReLU等传统激活函数参数量为$2sd$。
使用$SwiGLU$之后，设中间维度为 $x$ ，总参数量为 $3sx$ 。若要满足总参数量相等：
$$2sd = 3sx \\ \Rightarrow x = \frac{2}{3}d $$
也就是同样参数大小的模型使用SwiGLU之后中间隐层维度会变窄为原来$\frac{2}{3}$ 。