#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: BFGS.py
@time: 2019/2/26 15:43
"""

import numpy as np
import numpy as np

#函数表达式
fun = lambda x:100*(x[0]**2 - x[1]**2)**2 +(x[0] - 1)**2

#梯度向量
gfun = lambda x:np.array([400*x[0]*(x[0]**2 - x[1]) + 2*(x[0] - 1),-200*(x[0]**2 - x[1])])

#Hessian矩阵
hess = lambda x:np.array([[1200*x[0]**2 - 400*x[1] + 2,-400*x[0]],[-400*x[0],200]])

def bfgs(fun,gfun,hess,x0):
    #功能：用BFGS族算法求解无约束问题：min fun(x) 优化的问题请参考文章开头给出的链接
    #输入：x0是初始点，fun,gfun分别是目标函数和梯度，hess为Hessian矩阵
    #输出：x,val分别是近似最优点和最优解,k是迭代次数
    maxk = 1e5
    rho = 0.55
    sigma = 0.4
    gama = 0.7
    epsilon = 1e-5
    k = 0
    n = np.shape(x0)[0]
    #海森矩阵可以初始化为单位矩阵
    Bk = np.eye(n) #np.linalg.inv(hess(x0)) #或者单位矩阵np.eye(n)

    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0*np.linalg.solve(Bk,gk)
        m = 0
        mk = 0
        while m < 20: # 用Wolfe条件搜索求步长
            gk1 = gfun(x0 + rho**m*dk)
            if fun(x0+rho**m*dk) < fun(x0)+sigma*rho**m*np.dot(gk,dk) and np.dot(gk1.T, dk) >=  gama*np.dot(gk.T,dk):
                mk = m
                break
            m += 1

        #BFGS校正
        x = x0 + rho**mk*dk
        print("第"+str(k)+"次的迭代结果为："+str(x))
        sk = x - x0
        yk = gfun(x) - gk

        if np.dot(sk,yk) > 0: # sk 2 * 1， Bk 2 * 2
            Bs = np.dot(Bk,sk)
            ys = np.dot(yk,sk)
            sBs = np.dot(np.dot(sk,Bk),sk)
            Bk = Bk - 1.0*Bs.reshape((n,1))*Bs/sBs + 1.0*yk.reshape((n,1))*yk/ys
            print(Bk)
        k += 1
        x0 = x

    return x0,fun(x0),k#分别是最优点坐标，最优值，迭代次数
x0 ,fun0 ,k = bfgs(fun,gfun,hess,np.array([3,3]))
print(x0,fun0,k)
