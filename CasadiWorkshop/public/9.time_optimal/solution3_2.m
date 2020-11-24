
close all
clear all
import casadi.*

% Physical constants

N = 20; % Number of control intervals

%%
% ----------------------------------
%    continuous system dot(x)=f(x,u)
% ----------------------------------
nx = 4;

% Construct a CasADi function for the ODE right-hand side
x = MX.sym('x',nx); % states: pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s]
u = MX.sym('u',2); % control force [N]
rhs = [x(3:4);u];

%%
% -----------------------------------
%    Discrete system x_next = F(x,u)
% -----------------------------------

T = MX.sym('T');

opts = struct;
opts.tf = 1;
intg = integrator('intg','cvodes',struct('x',x,'p',[u;T],'ode',T*rhs),opts);

%%
% ------------------------------------------------
% Waypoints
% ------------------------------------------------

ref = [sin(linspace(0,2,N+1));cos(linspace(0,2,N+1))];

%%
% -----------------------------------------------
%    Optimal control problem, multiple shooting
% -----------------------------------------------

opti = casadi.Opti();

% Decision variables for states
X = opti.variable(nx,N+1);
% Decision variables for control vector
U =  opti.variable(2,N); % force [N]

T = opti.variable();
% Gap-closing shooting constraints
for k=1:N
  res = intg('x0',X(:,k),'p',vertcat(U(:,k),T/N));
  opti.subject_to(X(:,k+1)==res.xf);
end

% Path constraints
opti.subject_to(-3  <= X(1,:) <= 3); % pos_x limits
opti.subject_to(-3  <= X(2,:) <= 3); % pos_y limits
opti.subject_to(-3  <= X(3,:) <= 3); % vel_x limits
opti.subject_to(-3  <= X(4,:) <= 3); % vel_y limits
opti.subject_to(-10 <= U(1,:) <= 10); % force_x limits
opti.subject_to(-10 <= U(2,:) <= 10); % force_x limits

% Initial constraints
opti.subject_to(X(:,1)==[ref(:,1);0;0]);

% Try to follow the waypoints
opti.minimize(sumsqr(X(1:2,:)-ref));

% Time is bounded
opti.subject_to(0.5<=T<=2);
opti.set_initial(T, 1);

opti.solver('ipopt');

sol = opti.solve();

%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------

xsol = sol.value(X);
usol = sol.value(U);

tgrid = linspace(0,sol.value(T),N+1);

figure
hold on
plot(xsol(1,:)',xsol(2,:)','bs-','linewidth',2)
plot(ref(1,:)',ref(2,:)','ro','linewidth',3)
legend('OCP trajectory','Reference trajecory')
title('Top view')
xlabel('x')
xlabel('y')
figure
stairs(tgrid,[usol usol(:,end)]')
title('applied control signal')
legend('force_x','force_y')
ylabel('Force [N]')
xlabel('Time [s]')
