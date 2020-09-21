---
title: Understanding Black-box Predictions via Influence Functions
tags: MachineLearning
mathjax: true
---

# 通过 Infuluence functions 来理解黑匣子的预测模型

## 方法
对于一个输入为$$\chi$$输出为y的训练集，有一些训练点$$z_1, ... ,z_n$$。其中$$z_i = (x_i, y_i) \in X \times Y $$。对于一个训练点 z 和参数 $$\theta \in \Theta$$，假设$$L(z,\theta)$$是损失函数,则$$\frac{1}{n}\sum_{i=1}^n L(z_i,\theta)$$是最小经验损失函数。$$\hat{\theta} \mathop{=}\limits^{def} \mathop{argmin}_{\theta \in \Theta}  \frac{1}{n}\sum_{i=1}^n L(z_i,\theta)$$。

假设经验风险函数是二阶可微且严格凸的，则有下列推导：

## 增加训练点的值

我们的目标是理解训练点对于一个预测模型的作用。通过使用使用反事实条件。如果拿掉某一训练点，最后得到的最优解是什么样的呢？

首先从训练集中移走某一个 $$z$$ 的训练点，这时 $$\theta$$ 改变的量为$$\hat{\theta}_{-z} - \hat{\theta} $$

其中

$$\hat{\theta}_{-z} \mathop{=}\limits^{def} \mathop{argmin}_{\theta \in \Theta}  \frac{1}{n}\sum_{z_i\not=z} L(z_i,\theta)$$

然而，每移除一个训练点z就重新做一次训练显得复杂又不切实际。

Influence functions 给了我们一个效率的近似方法。这个方法是计算增加了一个很小的 $$\varepsilon$$ 时，参数的变化。这时我们得到了一个新的参数 $$\hat{\theta}_{\varepsilon,z} \mathop{=}\limits^{def} \mathop{argmin}_{\theta \in \Theta}  \frac{1}{n}\sum_{i=1}^n L(z_i,\theta) + \varepsilon L(z,\theta)$$ ，经典结果告诉我们，对z添加的干扰对结果的影响可以由下式刻画：

$${\cal L}_{up,params}(z) \mathop{=}\limits^{def} \frac{\mathrm{d} \hat\theta_{\varepsilon,z} }{\mathrm{d} \varepsilon } \bigg|_{\varepsilon = 0} = -H_{\hat{\theta}}^{-1} \nabla_{\theta} L(z,\hat{\theta}) $$

其中$$H_{\hat{\theta}} \mathop{=}\limits^{def} \frac{1}{n}\sum_{i=1}^n \nabla_{\theta}^2 L(z_i,\hat{\theta}) $$是hessian矩阵，且假设正定。本质上，我们通过牛顿法对经验风险函数做了一个二次近似。这时候移除一个训练点 $z$ 等价于它增加一个$\varepsilon = -\frac{1}{n}$ 的增量。我们可以通过这个近似来计算$$ \hat{\theta}_{-z} - \hat{\theta} \approx -\frac{1}{n} {\cal L}_{up,params}(z) $$，从而不用再次训练模型。

接下来使用求导的链式法则来计算怎样增加 $ z $ 改变 $ \theta $ 的函数。特别的，增加 $ z $ 对函数的影响使用 $ z_{test} $ 可以有下式给出：

$$
 \begin{equation}
 \begin{aligned}
 {\cal L}_{up,loss}(z,z_{test}) & \mathop{=}\limits^{def} \frac{\mathrm{d} L(z_{test},\hat\theta_{\varepsilon,z}) }{\mathrm{d} \varepsilon } \bigg|_{\varepsilon = 0} \\
 & = \nabla_\theta L(z_{test},\hat\theta)^ \mathrm{ T } \frac{\mathrm{d} \hat\theta_{\varepsilon,z} }{\mathrm{d} \varepsilon } \bigg|_{\varepsilon = 0} \\
 & = -\nabla_\theta L(z_{test},\hat\theta)^ \mathrm{ T } H_{\hat\theta}^{-1} \nabla_{\theta}L(z,\hat\theta)
 \end{aligned}
 \end{equation}
$$

## 干扰训练输入

如果修改了一个训练输入，模型的预测会发生什么变化呢？

对于一个训练点 $ z = (x,y) $ ，定义 $ z_{\delta} \mathop{=}\limits^{def} (x + \delta,y) $ 考虑到 $ z \rightarrow z_{\delta} $ 且令 $ \hat\theta_{z_\delta,-z} $ 是当用$ z_\delta $ 代替 $ z $ 后的最小经验风险函数时 $ \theta $ 的取值。为了达到近似效果，定义参数 $ \epsilon $ 从 $ z $ 变化到 $ z_\delta $。


# Influence function 的推导过程

设$\hat{\theta}$ 是经验风险函数的最小值：

$$ R(\theta) \mathop{=}\limits^{def} \frac{1}{n}\sum_{i=1}^n L(z_i,\theta) $$

假设函数R是二阶可导且为强凸。则：

$$ H_{\hat{\theta}} \mathop{=}\limits^{def} \nabla^2 R(\hat{\theta}) = \frac{1}{n}\sum_{i=1}^n \nabla_{\theta}^2 L(z_i,\hat{\theta}) $$

存在且正定，这保证了 $ H_{\hat{\theta}}^{-1} $ 存在。

参数 $ \hat{\theta}_{\varepsilon,z} $ 可以写作：

$$ \hat{\theta}_{\varepsilon,z} = \mathop{argmin}_{\theta \in \Theta} \{ R(\theta)+\varepsilon L(z,\theta)  \} $$

定义 $ \Delta_{\varepsilon} = \hat{\theta}_{\varepsilon,z} - \hat{\theta} $。由于 $ \hat{\theta} $ 不依赖于 $ \varepsilon $ 因此有以下式子：

$$ \frac{\mathrm{d} \hat\theta_{\varepsilon,z} }{\mathrm{d} \varepsilon } = \frac{\mathrm{d} \Delta_{\varepsilon} }{\mathrm{d} \varepsilon }  $$

因为 $ \hat{\theta}_{\varepsilon,z} $ 是前式的最小值，则对前式求一阶导有：

$$ 0 = \nabla R(\hat{\theta}_{\varepsilon,z}) + \varepsilon \nabla L(z,\hat{\theta}_{\varepsilon,z}) $$

因为 $ \hat{\theta}_{\varepsilon,z} \rightarrow \hat{\theta} $ 且 $ \varepsilon \rightarrow 0 $ 则对右侧式子做 Talor 二阶展开，有：

$$ 0 \approx \left[ \nabla R(\hat{\theta}) + \varepsilon \nabla L(z,\hat{\theta}_{\varepsilon,z}) \right] + \left[ \nabla^2 R(\hat{\theta}) + \varepsilon \nabla^2 L(z,\hat{\theta}_{\varepsilon,z}) \right] \Delta_\varepsilon $$

其中，$ o(\|\| \Delta_\varepsilon \|\|) $ 舍去。

整理后得到：

$$ \Delta_\varepsilon \approx \left[ \nabla^2 R(\hat{\theta}) + \varepsilon \nabla^2 L(z,\hat{\theta}_{\varepsilon,z}) \right]^{-1} \left[ \nabla R(\hat{\theta}) + \varepsilon \nabla L(z,\hat{\theta}_{\varepsilon,z}) \right] $$

由于 $ \hat\theta $ 是函数 R 的最小值，因此 $ \nabla R(\hat\theta) = 0 $ ，因此有：

$$ \Delta_\varepsilon \approx - \nabla^2 R(\hat{\theta})^{-1} \nabla L(z,\hat\theta)\varepsilon $$

联立前式，得到：

$$ \frac{\mathrm{d} \hat\theta_{\varepsilon,z} }{\mathrm{d} \varepsilon } \bigg|_{\varepsilon = 0} = -H_{\hat{\theta}}^{-1} \nabla_{\theta} L(z,\hat{\theta}) \mathop{=}\limits^{def} {\cal L}_{up,params}(z) $$