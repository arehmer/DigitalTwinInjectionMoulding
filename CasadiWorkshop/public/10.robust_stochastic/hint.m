
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

