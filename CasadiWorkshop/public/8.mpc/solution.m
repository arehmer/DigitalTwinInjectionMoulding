
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


speedup = 4; % 0..4

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
if speedup>=3
    intg_options.simplify = true;
end
intg_options.tf = dt/intg_options.number_of_finite_elements;

% Reference Runge-Kutta implementation
intg = integrator('intg','rk',struct('x',x,'p',u,'ode',f(x,u)),intg_options);
res = intg('x0',x,'p',u);
xf = res.xf;

% Discretized (sampling time dt) system dynamics as a CasADi Function
F = Function('F', {x, u}, {xf});

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

% 1.2: Parameter for initial state
x0 = opti.parameter(nx);

% Gap-closing shooting constraints
for k=1:N
   opti.subject_to(X(:,k+1)==F(X(:,k),U(k)));
end

% Path constraints
opti.subject_to(-3  <= pos <= 3);
opti.subject_to(-1.2 <= U   <= 1.2);

% Initial and terminal constraints
opti.subject_to(X(:,1)==x0);
opti.subject_to(X(:,N+1)==[0;0;0;0]);

% Objective: regularization of controls
% 1.1: added regularization
opti.minimize(sumsqr(U)+1000*sumsqr(pos));

% solve optimization problem
options = struct;
options.print_time = false;

if speedup>=3  
    options.expand = true; % expand makes function evaluations faster but requires more memory
end

if speedup>=2
    options.qpsol = 'qrqp';
    options.qpsol_options.print_iter = false;
    options.qpsol_options.print_header = false;
    options.print_iteration = false;
    options.print_header = false;
    options.print_status = false;
    opti.solver('sqpmethod',options)
else
    options.ipopt.print_level = 0;
    opti.solver('ipopt',options)
end

opti.set_value(x0,[0.5;0;0;0]);

sol = opti.solve();
%%
% -----------------------------------------------
%    MPC loop
% -----------------------------------------------

current_x = [0.5;0;0;0];

x_history = zeros(nx,400);
u_history = zeros(1,400);

rand ("state", 0)

disp('MPC running')

if speedup<4
  
  for i=1:400
      tic
      % What control signal should I apply?
      u_sol = sol.value(U(1));
      
      u_history(:,i) = u_sol;
      
      % Simulate the system over dt
      current_x = full(F(current_x,u_sol));
      if i>200
          current_x = current_x + [0;0;0;0.01*rand];
      end

      if speedup>=1
        % Set the initial values to the previous results
        opti.set_initial(opti.x, sol.value(opti.x)); % decision variables
        opti.set_initial(opti.lam_g, sol.value(opti.lam_g)); % multipliers
      end
    
      % Set the value of parameter x0 to the current x
      opti.set_value(x0, current_x);

      % Solve the NLP
      sol = opti.solve();
      
      x_history(:,i) = current_x;
      toc
  end

end

if speedup>=4
  inputs = {x0,opti.x,opti.lam_g};
  outputs = {U(1),opti.x,opti.lam_g};
  mpc_step = opti.to_function('mpc_step',inputs,outputs)
  
  u = sol.value(U(1));
  x = sol.value(opti.x);
  lam = sol.value(opti.lam_g);
  
  for i=1:400
      tic
      u_history(:,i) = full(u);
      
      % Simulate the system over dt
      current_x = full(F(current_x,u));
      if i>200
          current_x = current_x + [0;0;0;0.01*rand];
      end

      [u,x,lam] = mpc_step(current_x,x,lam);
      
      x_history(:,i) = current_x;
      toc
  end
  
end


%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------

figure
plot(x_history')
title('simulated states')
xlabel('sample')
figure
plot(u_history)
title('applied control signal')
ylabel('Force [N]')
xlabel('sample')

