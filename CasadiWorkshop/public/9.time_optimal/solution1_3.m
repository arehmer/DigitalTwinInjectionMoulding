
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
dt = MX.sym('dt');
xf = x+dt*f(x,u);
F = Function('F', {x, u, dt}, {xf});

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
T =  opti.variable(); % Time [s]

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
opti.subject_to(X(:,1)==[ref(:,1);0;0]);

% Time is bounded
opti.subject_to(0.5<=T<=2);
opti.set_initial(T, 1);

% Try to follow the waypoints
opti.minimize(sumsqr(X(1:2,:)-ref));

%opti.solver('ipopt');
opti.solver('sqpmethod',struct('convexify_strategy','eigen-clip'));

sol = opti.solve();

% 1.3: More than one step since problem is no longer a QP:
%      The gap-closing constraint just became nonlinear (quadratic)
%      in X,U,T -> we're not in Kansas anymore.
%
%      Worse, IPOPT shows that it needs to apply some regularization.
%      Free end-time problems are fundamentally harder than fixed end-time problems

figure()
spy(jacobian(opti.g,opti.x))

% 1.3 bis: the system dynamics depends on T for any control interval

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