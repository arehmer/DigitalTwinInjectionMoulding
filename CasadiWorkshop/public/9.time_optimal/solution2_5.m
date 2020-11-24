
close all
clear all
import casadi.*

% Physical constants

N = 40; % Number of control intervals


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

dt = MX.sym('dt'); % length of 1 control interval [s]

k1 = f(x, u);
k2 = f(x + dt/2 * k1, u);
k3 = f(x + dt/2 * k2, u);
k4 = f(x + dt * k3, u);
xf = x+dt/6*(k1 +2*k2 +2*k3 +k4);

F = Function('F', {x, u, dt}, {xf});

%%
% -----------------------------------------------
%    Optimal control problem, multiple shooting
% -----------------------------------------------

opti = casadi.Opti();

% Decision variables for states
X = opti.variable(nx,N+1);
% Decision variables for control vector
U =  opti.variable(2,N); % force [N]

% Time
T1 = opti.variable(); % Time scale for first half of trajectory
T2 = opti.variable(); % Time scale for second half of trajectory

% Gap-closing shooting constraints
for k=1:N/2
  opti.subject_to(X(:,k+1)==F(X(:,k),U(:,k),2*T1/N));
end
for k=N/2+1:N
  opti.subject_to(X(:,k+1)==F(X(:,k),U(:,k),2*T2/N));
end

% Path constraints
opti.subject_to(-3  <= X(1,:) <= 3); % pos_x limits
opti.subject_to(-3  <= X(2,:) <= 3); % pos_y limits
opti.subject_to(-3  <= X(3,:) <= 3); % vel_x limits
opti.subject_to(-3  <= X(4,:) <= 3); % vel_y limits
opti.subject_to(-10 <= U(1,:) <= 10); % force_x limits
opti.subject_to(-10 <= U(2,:) <= 10); % force_x limits

% Initial constraints
opti.subject_to(X(1:2,1)==[0;0]);

opti.subject_to(X(1:2,N/2)==[2;1]);

opti.subject_to(X(1:2,N+1)==[0;0]);
opti.subject_to(X(3:4,N+1)==X(3:4,1));

% Obstacle avoidance
p = [1;0.75];
r = 0.6;

opti.subject_to(sum((X(1:2,:)-p).^2,1)>=r^2);

% Time is bounded
opti.subject_to(0.5<=T1<=3);
opti.set_initial(T1, 0.5);
opti.set_initial(T1, 0.5);
opti.subject_to(0.5<=T2<=3);

opti.set_initial(X(1,:),cos(-linspace(0,2*pi,N+1)-pi)+p(1));
opti.set_initial(X(2,:),sin(-linspace(0,2*pi,N+1)-pi)+p(2));

% Try to follow the waypoints
opti.minimize(T1+T2);

opti.solver('ipopt');

sol = opti.solve();

sol.value(T1+T2) % 1.9861

%%
% -----------------------------------------------
%    Post-processing: plotting
% ----------------------------------------------

xsol = sol.value(X);
usol = sol.value(U);

tgrid1 = linspace(0,sol.value(T1),N/2+1);
tgrid2 = linspace(0,sol.value(T2),N/2+1);

tgrid = [tgrid1 tgrid1(end)+tgrid2(2:end)];

figure
hold on
plot(xsol(1,:)',xsol(2,:)','bs-','linewidth',2)
plot(xsol(1,1:N/2)',xsol(2,1:N/2)','bs-','linewidth',4)
plot(2,1,'ro')
t = linspace(0,2*pi,1000);
plot(r*sin(t)+p(1),r*cos(t)+p(2))
legend('OCP trajectory','obstacle')
title('Top view')
axis('equal')
xlabel('x')
xlabel('y')
figure
stairs(tgrid,[usol usol(:,end)]')
title('applied control signal')
legend('force_x','force_y')
ylabel('Force [N]')
xlabel('Time [s]')