close all
clear all
import casadi.*

T = 1; % control horizon [s]
N = 40; % Number of control intervals

dt = T/N; % length of 1 control interval [s]

%%
% ----------------------------------
%    continuous system dot(x)=f(x,u)
% ----------------------------------

% Construct a CasADi function for the ODE right-hand side

A = [0 0 0 0 0 0 1 0;
     0 0 0 0 0 1 0 1;
     0 0 0 0 1 0 0 1;
     0 0 0 1 0 0 0 1;
     0 0 1 1 1 1 1 1;
     0 1 0 0 0 0 0 1;
     1 0 0 0 0 0 0 1;
     1 1 1 0 0 0 1 1];
nx = size(A,1);
B = [  -0.073463  -0.073463;
  -0.146834  -0.146834;
  -0.146834  -0.146834;
  -0.146834  -0.146834;
  -0.446652  -0.446652;
  -0.147491  -0.147491;
  -0.147491  -0.147491;
  -0.371676  -0.371676];

nu = size(B,2);

x  = MX.sym('x',nx);
u  = MX.sym('u',nu);

dx = sparse(A)*sqrt(x)+sparse(B)*u;

x_steady = (-A\B*[1;1]).^2;

% Continuous system dynamics as a CasADi Function
f = Function('f', {x, u}, {dx});

%%
% -----------------------------------------------
%    Optimal control problem, multiple shooting
% -----------------------------------------------

opti = casadi.Opti();

% Decision variables for states
X = opti.variable(nx,N+1);
% Decision variables for control vector
U =  opti.variable(nu,N);

% Gap-closing shooting constraints
for k=1:N
   x = X(:,k);
   u = U(:,k);
   k1 = f(x, u);
   k2 = f(x + dt/2 * k1, u);
   k3 = f(x + dt/2 * k2, u);
   k4 = f(x + dt * k3, u);
   xf = x+dt/6*(k1 +2*k2 +2*k3 +k4);
   opti.subject_to(X(:,k+1)==xf);
end

% Path constraints
opti.subject_to(0.01 <= X(:) <= 0.1);

% Initial guesses
opti.set_initial(X, repmat(x_steady,1,N+1));
opti.set_initial(U, 1);

% Initial and terminal constraints
opti.subject_to(X(:,1)==x_steady);
% Objective: regularization of controls

xbar = opti.variable();
opti.minimize(1e-6*sumsqr(U)+sumsqr(X(:,N+1)-xbar));

% solve optimization problem
opti.solver('ipopt')

sol = opti.solve();

% There are dense nx-by-nx blocks here

figure()
spy(sol.value(jacobian(opti.g,opti.x)))

%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------

figure()
Xsol = sol.value(X);
plot(Xsol','o-')
