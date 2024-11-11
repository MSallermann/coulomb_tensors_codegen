# coulomb_tensors_codegen
Code generation for cartesian coulomb tensors.

The Coulomb tensors are defined as

```math
T^{ij} = \frac{1}{|r_i - r_j|}\lambda_0(r) = \frac{1}{r}\lambda_0(r),
```

where $\lambda_0$ is a short ranged switching function.

The gradients act to increase the rank of the screened interaction tensors

```math
\nabla_\alpha T^{ij} = T^{ij}_\alpha = -\frac{r_\alpha}{r^3}\lambda_1(r)
```

and

```math
\nabla_\alpha T^{ij}_\alpha = T^{ij}_{\alpha \beta} = 3\frac{r_{\alpha}r_{\beta}}{r^5} \lambda_2(r) - \frac{\delta_{\alpha \beta}}{r^3}\lambda_1(r),
```

where $r_\alpha = (r_j - r_i)_\alpha$.


In general the rank $n$ tensor $T^{ij}_{\alpha_0 \alpha_1..\alpha_n}$ can be written as

```math
T_{\alpha_1 .. \alpha_n} = 
\sum_{l=\lfloor \frac{n+1}{2} \rfloor}^{n}(-1)^{l}
\frac{(2l-1)!!}{R^{2l+1}} 
\sum_{\sigma} r_{\sigma(\alpha_1)} ... r_{\sigma(\alpha_{2l-n})}  \delta_{ \sigma(\alpha_{2l-n + 1})\sigma(\alpha_{2l-n+2}) } ... \delta_{ \sigma(\alpha_{n-1})\sigma(\alpha_n) }
```

where $\sigma$ runs over all permutations that __lead to a new summand__.

Explicitly, this means $\sigma$ runs over all permutations of $\alpha_1, ..., \alpha_n$ which do __not__
- swap the position of two $r$-factors, because $r_\alpha r_\beta = r_\beta r_\alpha$
- swap the position of two $\delta$ functions, because $\delta_{\alpha \beta}\delta_{\gamma\delta} = \delta_{\gamma\delta}\delta_{\alpha \beta}$
- swap the position of two $\delta$ indices, because $\delta_{\alpha \beta} = \delta_{\beta \alpha}$
- do not swap the position of an $r$ and a $\delta$, because $r_\alpha \delta_{\beta\gamma} = \delta_{\beta\gamma}r_\alpha$