
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

ref = @(s) [sin(2*s);cos(2*s)];

%%
% -----------------------------------------------
%    Optimal control problem, multiple shooting
% -----------------------------------------------

opti = casadi.Opti();

% Decision variables for states
X = opti.variable(nx,N+1);
% Decision variables for control vector
U =  opti.variable(2,N); % force [N]

S = opti.variable(1,N+1);

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
opti.subject_to(X(:,1)==[0;1;0;0]);

opti.subject_to(X(1:2,N+1)==[sin(2);cos(2)]);

opti.subject_to(sum1((X(1:2,:)-ref(S)).^2)'<=0.2^2);

opti.set_initial(S, linspace(0,1,N+1));

% Try to follow the waypoints
opti.minimize(T);

% Time is bounded
opti.subject_to(0.5<=T<=2);
opti.set_initial(T, 1);

opti.solver('ipopt');

sol = opti.solve();

sol.value(T) % 0.66866

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

R = ref(linspace(0,1,100));
plot(R(1,:)',R(2,:)','r','linewidth',3)
circle_x = 0.2*cos(linspace(0,2*pi));
circle_y = 0.2*sin(linspace(0,2*pi));
for i=1:100
  plot(R(1,i)+circle_x,R(2,i)+circle_y,'r.','linewidth',3)
end
legend('OCP trajectory','Reference trajecory')
title('Top view')
xlabel('x')
xlabel('y')
axis('equal')
figure
stairs(tgrid,[usol usol(:,end)]')
title('applied control signal')
legend('force_x','force_y')
ylabel('Force [N]')
xlabel('Time [s]')
