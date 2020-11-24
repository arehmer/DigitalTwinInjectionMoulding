clear all
close all

import casadi.*
%%

N = 25;
m = 40/N; % mass [kg]
D = 70*N; % spring constant [J/m^2]
g = 9.81; % gravitational constant [m/s^2]
L = 5/N;  % reference length [m]

%% 1.1

opti = casadi.Opti();

x = opti.variable(N);
y = opti.variable(N);

V = 0.5*D*sum(diff(x).^2+diff(y).^2);
V = V + g*sum(m*y);

opti.minimize(V);
opti.subject_to([x(1);y(1)] == [-2;0]);
opti.subject_to([x(N);y(N)] == [2;0]);
opti.solver('ipopt');

sol = opti.solve();

plot(sol.value(x),sol.value(y),'-o');

%% 1.2
spy(hessian(opti.f,opti.x))

% We observe a band structure: there is only coupling between immediately neighbouring points in the objective.
% We also note that there is no coupling between x and y.

%% 1.3

lag = opti.f+opti.lam_g'*opti.g;

sol.value(gradient(lag,opti.x))

norm(ans)

%% 2.1
opti = casadi.Opti();

x = opti.variable(N);
y = opti.variable(N);

V = 0.5*D*sum((sqrt(diff(x).^2+diff(y).^2)-L).^2);
V = V + g*sum(m*y);

opti.minimize(V);
opti.subject_to([x(1);y(1)] == [-2;0]);
opti.subject_to([x(N);y(N)] == [2;0]);
opti.solver('ipopt');

try
  sol = opti.solve(); % raises error
catch exception
  try
    disp(exception.getReport) 
  catch
    disp(exception.message)
  end
end

%% 2.1 (bis)

opti = casadi.Opti();

x = opti.variable(N);
y = opti.variable(N);

V = 0.5*D*sum((sqrt(diff(x).^2+diff(y).^2)-L).^2);
V = V + g*sum(m*y);

opti.minimize(V);
opti.subject_to([x(1);y(1)] == [-2;0]);
opti.subject_to([x(N);y(N)] == [2;0]);
opti.solver('ipopt');

opti.set_initial(x,linspace(-2,2,N)); % needed to avoid NaN

sol = opti.solve();

plot(sol.value(x),sol.value(y),'-o');
axis equal
spy(hessian(opti.f,opti.x))

%% 2.2


opti = casadi.Opti();

z = opti.variable(2*N,1);
x = z(1:2:end-1);
y = z(2:2:end);

V = 0.5*D*sum((sqrt(diff(x).^2+diff(y).^2)-L).^2);
V = V + g*sum(m*y);

opti.minimize(V);
opti.subject_to([x(1);y(1)] == [-2;0]);
opti.subject_to([x(N);y(N)] == [2;0]);
opti.solver('ipopt');

opti.set_initial(x,linspace(-2,2,N)); % needed to avoid NaN

sol = opti.solve();

plot(sol.value(x),sol.value(y),'-o');
axis equal

spy(hessian(opti.f,opti.x))

%% 2.3

opti = casadi.Opti();

x = opti.variable(N);
y = opti.variable(N);

V = 0.5*D*sum((sqrt(diff(x).^2+diff(y).^2)-L).^2);
V = V + g*sum(m*y);

opti.minimize(V);
opti.subject_to([x(1);y(1)] == [-2;0]);
opti.subject_to([x(N);y(N)] == [2;0]);
opti.solver('ipopt');

try
  opti.callback(@(i) plot_iteration(i,opti.debug.value(x),opti.debug.value(y)));
catch
  disp('callbacks not yet supported in octave');
end
opti.set_initial(x,linspace(-2,2,N)); % needed to avoid NaN

sol = opti.solve();

plot(sol.value(x),sol.value(y),'-o');
axis equal

%% 3.1
opti = casadi.Opti();

x = opti.variable(N);
y = opti.variable(N);

opti.minimize(g*sum(m*y));

opti.subject_to(diff(x).^2+diff(y).^2==L^2);
opti.subject_to([x(1);y(1)]==[-2;0]);
opti.subject_to([x(N);y(N)]==[2;0]);

opti.set_initial(x,linspace(-2,2,N));
opti.set_initial(y,-sin(linspace(0,pi,N)));

opti.solver('ipopt');

sol = opti.solve();

J = jacobian(opti.g,opti.x);
rank(full(opti.debug.value(J)))

plot(sol.value(x),sol.value(y),'-o');
axis equal
%% 3.2

close all
opti = casadi.Opti();

x = opti.variable(N);
y = opti.variable(N);

opti.minimize(g*sum(m*y));

opti.subject_to(diff(x).^2+diff(y).^2==L^2);
opti.subject_to([x(1);y(1)]==[-2;0]);
opti.subject_to([x(N);y(N)]==[2;0]);

opti.set_initial(x,linspace(-2,2,N));

opti.solver('ipopt');

try
  sol = opti.solve(); % Raises error
catch exception
  try
    disp(exception.getReport)
  catch
    disp(exception.message)
  end
end
J = jacobian(opti.g,opti.x);
size(J)
rank(full(opti.debug.value(J))) % Check LICQ: rank is 27 in stead of 28.
