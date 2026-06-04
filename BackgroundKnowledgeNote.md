# 背景知识笔记

1. 什么是 automatic differentiation?
    数学上, 微分即求导. 自动微分

2. 计算导数有哪些方法? 其中自动微分采用哪种方法? 有什么优势?

3. pytorch 中自动微分的原理是什么?

4. 什么是张量(tensor)?
    是支持自动微分, 可以在GPU加速计算的多维数组的统称. 例如, 0阶张量(rank-0 tensor)是标量(scalar), 1阶张量是向量(vector), 2阶张量是矩阵(matrix)

5. 为什么 tensor 支持 GPU 加速, numpy 的 ndarray 不行?

6. tensor 对象的 GPU support(硬件加速) 和 autograd engine(携带计算图) 是如何实现的?