clc
clear all
close all

import casadi.*

N = 100;

D = load('data.mat','-ascii');


p_hat = [1.43;0.99;0.22;0.022;0.022;0.011];

a = p_hat(1);
b = p_hat(2);
alpha = p_hat(3);
beta = p_hat(4);
gamma = p_hat(5);
delta = p_hat(6);

%% 1.1


x = 10;
y = 0.1;

for i=1:10
    x_next = (a*x-alpha*x*y)/(1+gamma*x);
    y_next = (b*y+beta*x*y)/(1+delta*y);

    x = x_next;
    y = y_next;
end


[x y] % Should be [13.1929 1.2829]

%% 1.2

e = 0;

x = 10;
y = 0.1;

for i=1:10
    e = e + (x-D(i,1))^2+(y-D(i,2))^2;
    
    x_next = (a*x-alpha*x*y)/(1+gamma*x);
    y_next = (b*y+beta*x*y)/(1+delta*y);

    x = x_next;
    y = y_next;
end
e = e + (x-D(11,1))^2+(y-D(11,2))^2;

e % Should be 56.893

%% 1.3

N = 100;

opti = Opti();

a = opti.variable();
b = opti.variable();
alpha = opti.variable();
beta = opti.variable();

gamma = opti.variable();
delta = opti.variable();

p = [a;b;alpha;beta;gamma;delta];

x = 10;
y = 0.1;

f = 0;

for i=1:N
    f = f + (x-D(i,1))^2+(y-D(i,2))^2;
    x_next = (a*x-alpha*x*y)/(1+gamma*x);
    y_next = (b*y+beta*x*y)/(1+delta*y);

    x = x_next;
    y = y_next;
end

f = f + (x-D(101,1))^2+(y-D(101,2))^2;

opti.minimize(f);

opti.solver('ipopt')

opti.set_initial(p,p_hat);

sol = opti.solve();

sol.value(p)' %   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]

%% 2.1


opti = Opti();

a = opti.variable();
b = opti.variable();
alpha = opti.variable();
beta = opti.variable();

gamma = opti.variable();
delta = opti.variable();

p = [a;b;alpha;beta;gamma;delta];

x = 10;
y = 0.1;

F = {};

for i=1:N
    F{end+1} = [x-D(i,1);y-D(i,2)];
    x_next = (a*x-alpha*x*y)/(1+gamma*x);
    y_next = (b*y+beta*x*y)/(1+delta*y);

    x = x_next;
    y = y_next;
end

F{end+1} = [x-D(101,1);y-D(101,2)];

F = vertcat(F{:});

opti.minimize(F'*F);

opti.solver('ipopt')

opti.set_initial(p,p_hat);

sol = opti.solve();

sol.value(p) %   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]


%% 2.2

opti = Opti();

a = opti.variable();
b = opti.variable();
alpha = opti.variable();
beta = opti.variable();

gamma = opti.variable();
delta = opti.variable();

p = [a;b;alpha;beta;gamma;delta];

x = 10;
y = 0.1;

F = {};

for i=1:N
    F{end+1} = [x-D(i,1);y-D(i,2)];
    x_next = (a*x-alpha*x*y)/(1+gamma*x);
    y_next = (b*y+beta*x*y)/(1+delta*y);

    x = x_next;
    y = y_next;
end

F{end+1} = [x-D(101,1);y-D(101,2)];

F = vertcat(F{:});

opti.minimize(F'*F);

J = jacobian(F,p);

H = 2*J'*J; % Factor two to be consistent with missing 0.5 in objective.

sigma = MX.sym('sigma');
opts.hess_lag = Function('hess_lag',{opti.x,opti.p,sigma,opti.lam_g}, {sigma*triu(H)});
opti.solver('ipopt',opts)

opti.set_initial(p,p_hat);

sol = opti.solve();

sol.value(p) %   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]

%% 2.3



opti = Opti();

a = opti.variable();
b = opti.variable();
alpha = opti.variable();
beta = opti.variable();

gamma = opti.variable();
delta = opti.variable();

p = [a;b;alpha;beta;gamma;delta];

x = 10;
y = 0.1;

S = {};

for i=1:N
    S{end+1} = [x y];
    x_next = (a*x-alpha*x*y)/(1+gamma*x);
    y_next = (b*y+beta*x*y)/(1+delta*y);

    x = x_next;
    y = y_next;
end

S{end+1} = [x y];

S = vertcat(S{:});

S = Function('S',{p},{S});

F = vec(S(p)-D);

opti.minimize(F'*F);

opti.set_initial(p,p_hat);

opti.solver('ipopt');
sol = opti.solve();

sol.value(p) %   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]



%% 2.4

D_modified = D;

D_modified(10,1) = 1;
F = vec(S(p)-D_modified);

opti.minimize(F'*F);
sol = opti.solve();

p_fit = sol.value(p)

D_fit = full(S(p_fit));

figure()
hold on
plot(D_fit)
plot(D_modified)

% That's quite a bad fit around the first bump

%% 2.5

opti = Opti();
p = opti.variable(6,1);
L = opti.variable(2*N+2);
F = vec(S(p)-D_modified);

opti.minimize(sum(L));
opti.subject_to(-L<=F<=L);

opti.solver('ipopt');

opti.set_initial(p,p_hat);

sol = opti.solve();

p_fit = sol.value(p) %   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]

D_fit = full(S(p_fit));

figure()
hold on
plot(D_fit)
plot(D_modified)
