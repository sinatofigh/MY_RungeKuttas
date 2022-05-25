#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as SCP


def my_lorenz_fun(t,Y,sig=10,rho=28,beta=8/3):
    x,y,z = Y

    dxdt = sig*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y - beta*z

    dYdt = np.array([dxdt,dydt,dzdt])

    return dYdt

def my_RK23_fun(fun,y0,t0,dt,N):
    num_dim = len(y0)
    Y = np.zeros((num_dim,N+1))
    Y[:,0] = y0
    t = 0
    for i in range(1,N+1):
        fk = Y[:,i-1] + (dt/2)*fun(t,Y[:,i-1])
        Y[:,i] = Y[:,i-1] + dt*fun(t,fk)
    return Y

def my_RK45_fun(fun,y0,t0,dt,N):
    num_dim = len(y0)
    Y = np.zeros((num_dim,N+1))
    Y[:,0] = y0
    t = np.arange(t0,t0+(N+1)*dt,dt)
    for k in range(1,N+1):
        f1 = fun(t[k-1], Y[:,k-1])
        f2 = fun(t[k-1]+(dt/2), Y[:,k-1]+(dt/2)*f1)
        f3 = fun(t[k-1]+(dt/2), Y[:,k-1]+(dt/2)*f2)
        f4 = fun(t[k-1]+dt, Y[:,k-1]+dt*f3)

        Y[:,k] = Y[:,k-1] + (dt/6)*(f1+2*f2+2*f3+f4)

    return Y

N = 2000
dt = 0.01
t0 = 0
Y0 = [-8,8,27]

my_sol_23 = my_RK23_fun(my_lorenz_fun,Y0, t0, dt, N)
my_sol_45 = my_RK45_fun(my_lorenz_fun,Y0,t0,dt,N)

scp_sol_23 = SCP.solve_ivp(my_lorenz_fun,(t0,t0+N*dt),Y0,method='RK23')
scp_sol_45 = SCP.solve_ivp(my_lorenz_fun,(t0,t0+N*dt),Y0,method='RK45')

s_min_23 = np.min([my_sol_23.shape[1],scp_sol_23.y.shape[1]])
ER_23 = np.abs(my_sol_23[:,0:s_min_23] - scp_sol_23.y[:,0:s_min_23])
rel_ER_23 = ER_23/np.abs(scp_sol_23.y[:,0:s_min_23])

s_min_45 = np.min([my_sol_45.shape[1],scp_sol_45.y.shape[1]])
ER_45 = np.abs(my_sol_45[:,0:s_min_45] - scp_sol_45.y[:,0:s_min_45])
rel_ER_45 = ER_45/np.abs(scp_sol_45.y[:,0:s_min_45])


plt.figure(num=0)
ax = plt.axes(projection='3d')
a1 = ax.plot3D(my_sol_23[0,:],my_sol_23[1,:],my_sol_23[2,:],'-b',label='my_solution_RK23')
a2 = ax.plot3D(scp_sol_23.y[0,:],scp_sol_23.y[1,:],scp_sol_23.y[2,:],'--r',label='scipy_solution_RK23')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()


plt.figure(num=1)
plt.plot(np.arange(s_min_23),rel_ER_23[0,:],'-b',label='relative error along x-axis')
plt.plot(np.arange(s_min_23),rel_ER_23[1,:],'-r',label='relative error along y-axis')
plt.plot(np.arange(s_min_23),rel_ER_23[2,:],'-g',label='relative error along z-axis')
plt.legend()
plt.grid(True)
plt.title(r"$\frac{|my solution RK23 - scipy solution RK23|}{|scipy solution RK23|}$")
plt.xlabel('iteration')
plt.ylabel('relative error')


plt.figure(num=2)
ax = plt.axes(projection='3d')
a1 = ax.plot3D(my_sol_45[0,:],my_sol_45[1,:],my_sol_45[2,:],'-b',label='my_solution_RK45')
a2 = ax.plot3D(scp_sol_45.y[0,:],scp_sol_45.y[1,:],scp_sol_45.y[2,:],'--r',label='scipy_solution_RK45')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()


plt.figure(num=3)
plt.plot(np.arange(s_min_45),rel_ER_45[0,:],'-b',label='relative error along x-axis')
plt.plot(np.arange(s_min_45),rel_ER_45[1,:],'-r',label='relative error along y-axis')
plt.plot(np.arange(s_min_45),rel_ER_45[2,:],'-g',label='relative error along z-axis')
plt.legend()
plt.grid(True)
plt.title(r"$\frac{|my solution RK45 - scipy solution RK45|}{|scipy solution RK45|}$")
plt.xlabel('iteration')
plt.ylabel('relative error')

plt.show()