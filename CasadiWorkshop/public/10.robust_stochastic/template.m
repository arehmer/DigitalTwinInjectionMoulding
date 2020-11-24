
close all
clear all
import casadi.*

% Physical constants

T = 1; % control horizon [s]
N = 40; % Number of control intervals

dt = T/N; % length of 1 control interval [s]

tgrid = linspace(0,T,N+1);

%%
% ----------------------------------
%    continuous system dot(x)=f(x,u)
% ----------------------------------
nx = 2;

% Construct a CasADi function for the ODE right-hand side
x1 = MX.sym('x1');
x2 = MX.sym('x2');
u = MX.sym('u'); % control
rhs = [x2;-0.1*(1-x1^2)*x2 - x1 + u];
x = [x1;x2];

x1_bound = @(t) 2+0.1*cos(10*t);

%%
% -----------------------------------
%    Discrete system x_next = F(x,u)
% -----------------------------------

opts = struct;
opts.tf = dt;
intg = integrator('intg','cvodes',struct('x',x,'p',u,'ode',rhs),opts);

%%
% -----------------------------------------------
%    Optimal control problem, multiple shooting
% -----------------------------------------------
x0 = [0.5;0];

opti = casadi.Opti();

% Decision variable for states
x = opti.variable(nx);

% Initial constraints
opti.subject_to(x==x0);

U = {};
X = {x};
% Gap-closing shooting constraints
for k=1:N
  u = opti.variable();
  U{end+1} = u;

  x_next = opti.variable(nx);
  res = intg('x0',x,'p',u);
  opti.subject_to(x_next==res.xf);

  opti.subject_to(-40<=u<=40);
  opti.subject_to(-0.25<=x(1)<=x1_bound(tgrid(k)));
  
  x = x_next;
  X{end+1} = x;
end
opti.subject_to(-0.25<=x_next(1)<=x1_bound(tgrid(N+1)));
U = [U{:}];
X = [X{:}];

opti.minimize(sumsqr(X(1,:)-3));

opti.solver('ipopt');

sol = opti.solve();

%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------


% Simulate forward in time using an initial state and control vector
usol = sol.value(U);
xsol = sol.value(X);


figure
hold on
plot(tgrid,xsol(1,:)','bs-','linewidth',2)
plot(tgrid,x1_bound(tgrid),'r--','linewidth',4)
legend('OCP trajectory x1','bound on x1')
xlabel('Time [s]')
ylabel('x1')
figure
stairs(tgrid,[usol usol(:,end)]')
title('applied control signal')
ylabel('Force [N]')
xlabel('Time [s]')