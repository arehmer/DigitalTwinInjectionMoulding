
close all
clear all
import casadi.*

% Physical constants

T = 1; % control horizon [s]
N = 20; % Number of control intervals

dt = T/N; % length of 1 control interval [s]

tgrid = linspace(0,T,N+1);

%%
% ----------------------------------
%    continuous system dot(x)=f(x,u)
% ----------------------------------
nx = 4;

% Construct a CasADi function for the ODE right-hand side
x = MX.sym('x',nx); % states: pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s]
u = MX.sym('u',2); % control force [N]
rhs = [x(3:4);u];

% Continuous system dynamics as a CasADi Function
f = Function('f', {x, u}, {rhs});


%%
% -----------------------------------
%    Discrete system x_next = F(x,u)
% -----------------------------------

k1 = f(x, u);
k2 = f(x + dt/2 * k1, u);
k3 = f(x + dt/2 * k2, u);
k4 = f(x + dt * k3, u);
xf = x+dt/6*(k1 +2*k2 +2*k3 +k4);

F = Function('F', {x, u}, {xf});

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

% Gap-closing shooting constraints
for k=1:N
  opti.subject_to(X(:,k+1)==F(X(:,k),U(:,k)));
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

opti.solver('ipopt');

sol = opti.solve();

%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------

xsol = sol.value(X);
usol = sol.value(U);

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