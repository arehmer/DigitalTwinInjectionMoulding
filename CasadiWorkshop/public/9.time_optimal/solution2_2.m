
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
T = opti.variable();

% Gap-closing shooting constraints
for k=1:N
  opti.subject_to(X(:,k+1)==F(X(:,k),U(:,k),T/N));
end

% Path constraints
opti.subject_to(-3  <= X(1,:) <= 3); % pos_x limits
opti.subject_to(-3  <= X(2,:) <= 3); % pos_y limits
opti.subject_to(-3  <= X(3,:) <= 3); % vel_x limits
opti.subject_to(-3  <= X(4,:) <= 3); % vel_y limits
opti.subject_to(-10 <= U(1,:) <= 10); % force_x limits
opti.subject_to(-10 <= U(2,:) <= 10); % force_x limits

% Initial constraints
opti.subject_to(X(:,1)==[0;0;0;0]);

opti.subject_to(X(1:2,N+1)==[2;1]);

% Obstacle avoidance
p = [1;0.75];
r = 0.6;

g = sqrt(sum((X(1:2,:)-p).^2,1))>=r;
opti.subject_to(g);

% Time is bounded
opti.subject_to(0.5<=T<=2);
opti.set_initial(T, 1);

branch = 1;

if branch==1
  opti.set_initial(X(1,:),linspace(0,2,N+1)); % Leads to T 8.4867232e-01 - 8.4911618e-01 (r=0.601)
else
  opti.set_initial(X(2,:),linspace(0,2,N+1)); % Leads to T 1.0437847e+00 - 1.0442997e+00 (r=0.601)
end

if branch==1
  dTdr = (8.4911618e-01-8.4867232e-01)/0.001 % 4.4386e-1
else
  dTdr = (1.0442997e+00-1.0437847e+00)/0.001 % 5.1500e-1
end

% Try to follow the waypoints
opti.minimize(T);

opti.solver('ipopt');

sol = opti.solve();

% There's only one constraint active; one (numerical) nonzero in the duals
if branch==1
  sol.value(opti.dual(g)) % for constraint with r^2: 3.6997e-01 - for constraint with r: 4.4397e-01 (matches with dTdr)
else
  sol.value(opti.dual(g)) % for constraint with r^2: 4.2918e-01 - for constraint with r: 5.1501e-01 (matches with dTdr)
end

% Note: when the constraint has r^2, the multiplier corresponds to dT^star/d(r^2) = dT^star/dr * d(r^2)/dr = dT^star/dr * 2 *r


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