clear all
close all
import casadi.*

%% 1.1


x = MX.sym('x',2);

f = x(1)^2+tanh(x(2))^2;
g = cos(sum(x))+0.5;
h = sin(x(1))+0.5;

%% 1.2

lambda = MX.sym('lambda'); % multiplier for g
nu = MX.sym('nu'); % multiplier for h

lag = f+lambda*g+nu*h;

lagf = Function('lagf',{x,lambda,nu},{lag});

lagf([-0.5;-1.8],2,3) % Should be 0.875613

%% 1.3


QPf = Function('QPf',{x,lambda,nu},{f,g,h,gradient(f,x),hessian(lag,x),jacobian(g,x),jacobian(h,x)});

%% 1.4

x0 = [-0.5;-1.8];
lambda0 = 0.;
nu0 = 0;

% Perform the NLP linearization
[nlp_f,nlp_g,nlp_h,nlp_grad_f,nlp_hess_l,nlp_jac_g,nlp_jac_h] = QPf(x0,lambda0,nu0);

DM.set_precision(16)

nlp_hess_l

%% 1.5

H = nlp_hess_l;
G = nlp_grad_f;
A = [nlp_jac_g;nlp_jac_h];
lba = [-nlp_g;-inf];
uba = [-nlp_g;-nlp_h];

disp(H)
disp(G)
disp(A)
disp(lba)
disp(uba)

%% 1.6

disp('H:')
H.sparsity()
spy(H)

figure()
disp('A:')
A.sparsity()
spy(A)


% The difference is between structural sparsity (displayed with 00),
% and sparsity by numerical coincidence.
% Indeed, change lambda0 or nu0 and the Hessian will become dense

%% 1.7
qp_struct = struct('h',sparsity(H),'a',sparsity(A));
solver = conic('solver','qrqp',qp_struct);
class(solver) % casadi.Function

%% 1.8
disp(solver)
res = solver('h',H,'g',G,'a',A,'lba',lba,'uba',uba)

dx = res.x % should be  [-0.02344447381848419, 0.2464226943869885]
lambda = res.lam_a(1) % should be 0.3785941041969475
nu = res.lam_a(2) % should be 0.871222132298292

%% 1.9

x = [-0.5;-1.8];
lambda = 0;
nu = 0;

opts = struct;
opts.print_iter = false;

solver = conic('solver','qrqp',qp_struct,opts);

for i=1:4

    % Compute linearizations
    [nlp_f,nlp_g,nlp_h,nlp_grad_f,nlp_hess_l,nlp_jac_g,nlp_jac_h] = QPf(x,lambda,nu);

    % Compose into matrices expected by solver
    H = nlp_hess_l;
    G = nlp_grad_f;
    A = [nlp_jac_g;nlp_jac_h];
    lba = [-nlp_g;-inf];
    uba = [-nlp_g;-nlp_h];

    % Call solver
    res = solver('h',H,'g',G,'a',A,'lba',lba,'uba',uba);

    % Interpret results
    dx = res.x;
    lambda = res.lam_a(1);
    nu = res.lam_a(2);
    
    x = x + dx

end

%% 2.1

% 4 equations, 4 unknowns

% rootfinder x:  [x;lambda;nu]
% rootfinder p: tau (something we may tune)
% rootfinder g: nu*h+tau

%% 2.2

x = MX.sym('x',2);

f = x(1)^2+tanh(x(2))^2;
g = cos(sum(x))+0.5;
h = sin(x(1))+0.5;

lambda = MX.sym('lambda');
nu = MX.sym('nu');

tau = MX.sym('tau');

lag = f+lambda*g+nu*h;

G = struct;
G.x = [x;lambda;nu];
G.p = tau;
G.g = [gradient(lag,x);g;nu*h+tau];

rf = rootfinder('rf','newton',G)

%% 2.3
x0 = [-0.5;-1.8];
lambda0 = 0.1;
nu0 = 0.1;

res = rf('x0',[x0;lambda0;nu0],'p',1e-2);
res.x(1:2)

%% 2.4
res = rf('x0',[x0;lambda0;nu0],'p',1e-6);
res.x(1:2)