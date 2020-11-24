close all
clear all
import casadi.*

% Physical constants
g = 9.81;    % gravitation [m/s^2]
L = 0.2;     % pendulum length [m]
m = 1;       % pendulum mass [kg]
mcart = 0.5; % cart mass [kg]

T = 2; % control horizon [s]
N = 160; % Number of control intervals

dt = T/N; % length of 1 control interval [s]

% System is composed of 4 states
nx = 4;

%%
% ----------------------------------
%    continuous system dot(x)=f(x,u)
% ----------------------------------

% Construct a CasADi function for the ODE right-hand side
x = MX.sym('x',nx); % states: pos [m], theta [rad], dpos [m/s], dtheta [rad/s]
u = MX.sym('u'); % control force [N]
ddpos = ((u+m*L*x(4)*x(4)*sin(x(2))-m*g*sin(x(2))*cos(x(2)))/(mcart+m-m*cos(x(2))*cos(x(2))));
rhs = [x(3); x(4) ;ddpos;  g/L*sin(x(2))-cos(x(2))*ddpos];

% Continuous system dynamics as a CasADi Function
f = Function('f', {x, u}, {rhs});

%%
% -----------------------------------
%    Discrete system x_next = F(x,u)
% -----------------------------------

% Integrator options
intg_options = struct;
intg_options.number_of_finite_elements = 1;
intg_options.tf = dt/intg_options.number_of_finite_elements;

% Reference Runge-Kutta implementation
intg = integrator('intg','rk',struct('x',x,'p',u,'ode',f(x,u)),intg_options);
res = intg('x0',x,'p',u);

% Discretized (sampling time dt) system dynamics as a CasADi Function
F = Function('F', {x, u}, {res.xf});

%%
% -----------------------------------------------
%    Optimal control problem, multiple shooting
% -----------------------------------------------

opti = casadi.Opti();

% Decision variables for states
X = opti.variable(nx,N+1);
% Aliases for states
pos    = X(1,:);
theta  = X(2,:);
dpos   = X(3,:);
dtheta = X(4,:);

% Decision variables for control vector
U =  opti.variable(N,1); % force [N]

% Gap-closing shooting constraints
for k=1:N
   opti.subject_to(X(:,k+1)==F(X(:,k),U(k)));
end

% Path constraints
opti.subject_to(-3  <= pos <= 3);
opti.subject_to(-1.2 <= U   <= 1.2);

% Initial and terminal constraints
opti.subject_to(X(:,1)==[1;0;0;0]);
opti.subject_to(X(:,N+1)==[0;0;0;0]);

% Objective: regularization of controls
opti.minimize(sumsqr(U));

% solve optimization problem
opti.solver('ipopt')

sol = opti.solve();

%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------

pos_opt = sol.value(pos);
theta_opt = sol.value(theta);
dpos_opt = sol.value(dpos);
dtheta_opt = sol.value(dtheta);

u_opt = sol.value(U);

% time grid for printing
tgrid = linspace(0,T, N+1);

figure;
subplot(3,1,1)
hold on
plot(tgrid, theta_opt, 'b')
plot(tgrid, pos_opt, 'b')
legend('theta [rad]','pos [m]')
xlabel('Time [s]')
subplot(3,1,2)
hold on
plot(tgrid, dtheta_opt, 'b')
plot(tgrid, dpos_opt, 'b')
legend('dtheta [rad/s]','dpos [m/s]')
xlabel('Time [s]')
subplot(3,1,3)
stairs(tgrid(1:end-1), u_opt, 'b')
legend('u [m/s^2]')
xlabel('Time [s]')

cart = [pos;0*pos];
ee   = [pos+L*sin(theta);L*cos(theta)];

cart_sol = sol.value(cart);
ee_sol   = sol.value(ee);

figure
hold on
for k=1:8:N+1
    line([cart_sol(1,k) ee_sol(1,k)],[cart_sol(2,k) ee_sol(2,k)],'LineWidth',1)
end
axis equal