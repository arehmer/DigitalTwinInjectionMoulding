
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

P = MX.sym('x',2,2);
rhs = [x2;-0.1*(1-x1^2)*x2 - x1 + u];
x = [x1;x2];

A = jacobian(rhs,x);

x = [x;P(:)];
dotP = A*P+P*A';

rhs = [rhs;dotP(:)];

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
P = opti.variable(nx,nx);
opti.set_initial(P,eye(2));

% Initial constraints
opti.subject_to(x==x0);
opti.subject_to(P==diag([0.01^2,0.1^2]));

U = {};
X = {x};
Ps = {P};
% Gap-closing shooting constraints
for k=1:N
  u = opti.variable();
  U{end+1} = u;

  x_next = opti.variable(nx);
  P_next = opti.variable(nx,nx);
  opti.set_initial(P_next,eye(2));
  res = intg('x0',[x;P(:)],'p',u);
  
  xf = res.xf(1:nx);
  Pf = reshape(res.xf(nx+1:end),nx,nx);
  opti.subject_to(x_next==xf);
  opti.subject_to(P_next(:)==Pf(:));

  opti.subject_to(-40<=u<=40);
  
  var = [1 0]*P*[1;0];
  sigma = sqrt(var);
  
  opti.subject_to(-0.25<=x(1)<=x1_bound(tgrid(k))-sigma);
  
  x = x_next;
  P = P_next;
  X{end+1} = x;
  Ps{end+1} = P;
end
var = [1 0]*P*[1;0];
sigma = sqrt(var);
  
opti.subject_to(-0.25<=x_next(1)<=x1_bound(tgrid(N+1))-sigma);
U = [U{:}];
X = [X{:}];

opti.minimize(sumsqr(X(1,:)-3));

opti.solver('ipopt');

sol = opti.solve();

spy(jacobian(opti.g,opti.x))
%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------
usol = sol.value(U);
xsol = sol.value(X);


figure
hold on
plot(tgrid,xsol(1,:)','bs-','linewidth',2)
plot(tgrid,x1_bound(tgrid),'r--','linewidth',4)

for k=1:N+1
  var = sol.value([1 0]*Ps{k}*[1;0]);
  sigma = sqrt(var);
  t = tgrid(k);
  plot([t,t],[xsol(1,k)-sigma,xsol(1,k)+sigma],'k','linewidth',2)
end
legend('OCP trajectory x1','bound on x1')
xlabel('Time [s]')
ylabel('x1')
figure
stairs(tgrid,[usol usol(:,end)]')
title('applied control signal')
ylabel('Force [N]')
xlabel('Time [s]')
