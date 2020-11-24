
close all
clear all
import casadi.*

% Physical constants

T = 1; % control horizon [s]
N = 6; % Number of control intervals

dt = T/N; % length of 1 control interval [s]

tgrid = linspace(0,T,N+1);

% The perturbation of delta
delta_num = 1;


%%
% ----------------------------------
%    continuous system dot(x)=f(x,u)
% ----------------------------------
nx = 2;

% Construct a CasADi function for the ODE right-hand side
x1 = MX.sym('x1');
x2 = MX.sym('x2');
u = MX.sym('u'); % control
delta = MX.sym('delta');
rhs = [x2;-0.1*(1-x1^2+delta)*x2 - x1 + u];
x = [x1;x2];

x1_bound = @(t) 2+0.1*cos(10*t);

%%
% -----------------------------------
%    Discrete system x_next = F(x,u)
% -----------------------------------

opts = struct;
opts.tf = dt;
intg = integrator('intg','cvodes',struct('x',x,'p',[u;delta],'ode',rhs),opts);

%% Traversing all branches

% The core of the algorithm will be a recursion that enumerates all
% possible branches/series of events

% Our implementation prodcues a cell with each entry corresponding to a
% unique sequence of events
H = recurse_dummy(delta_num,N,{});
vertcat(H{:})

% The actual 'recurse' function will return a cell of state variable
% sequences

%%
% -----------------------------------------------
%    Optimal control problem, multiple shooting
% -----------------------------------------------
x0 = [2;0];

opti = casadi.Opti();

% Decision variable for states
x = opti.variable(nx);

% Initial constraints
opti.subject_to(x==x0);

% Instead of true non-anticipativity constraints,
% we just have one common control for all.
U = opti.variable(1,N);
opti.subject_to(-40<=U'<=40);

% Sample the state bounds
x1_bound_sampled = x1_bound(tgrid);

% This function returns a cell with an entry for each possible series of events (=one full branch)
% Each cell entry has a concatenation of all states of that full branch.
X = recurse(opti,intg,delta_num,N,x,x1_bound_sampled(2:end),{x},U);

% We wish to minimize the maximum objective over all branches
% Instead of   min max(e1,e2,...), (cfr L-infinity)
% we write     min L
%                  s.t  L>=e1
%                       L>=e2
%                       ...
L = opti.variable();
for i=1:numel(X)
  x = X{i};
  opti.subject_to(L>=sumsqr(x(1,:)-3));
end
opti.minimize(L);

opti.solver('ipopt');

sol = opti.solve();

%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------


% Simulate forward in time using an initial state and control vector
usol = sol.value(U);


figure
hold on

% Draw each full branch
for i=1:numel(X)
  xsol = sol.value(X{i});
  norm(xsol(1,:)-3)^2
  plot(tgrid,xsol(1,:)','bs-','linewidth',2)
end

plot(tgrid,x1_bound(tgrid),'r--','linewidth',4)
xlabel('Time [s]')
ylabel('x1')
figure
stairs(tgrid,[usol usol(:,end)]')
title('applied control signal')
ylabel('Force [N]')
xlabel('Time [s]')
