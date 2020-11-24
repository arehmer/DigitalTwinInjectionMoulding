clear all
close all

%% 1.1

res = g([-0.8;2]) % Should be   [-0.2986;-0.7174]

%% 1.2

x = [-0.8;2];

nominal = g(x);
epsilon = 1e-8;
perturbed_x = (g(x+epsilon*[1;0])-nominal)/epsilon;
perturbed_y = (g(x+epsilon*[0;1])-nominal)/epsilon;

J = [perturbed_x perturbed_y] % Should be  [0.1457    0.1749; 0.6967   -0.5000]

%% 1.3

x = [-0.8;2];

for i=1:5
    nominal = g(x);
    epsilon = 1e-8;
    perturbed_x = (g(x+epsilon*[1;0])-nominal)/epsilon;
    perturbed_y = (g(x+epsilon*[0;1])-nominal)/epsilon;

    J = [perturbed_x perturbed_y];

    x = x - J\nominal; % newton step
end

x % Should be [0.1945;2.3866]

%% 2.2
import casadi.*
X = MX.sym('x',2);
g(X) % evaluate g symbolically

% @1=x[0], @2=x[1], vertcat(tanh(((((2+@1)*sq(@2))/25)-0.5)), (1+(sin(@1)-(0.5*@2))))
% You see tanh and sin applied to expressions involving components of the symbol x

class(g(X))
size(g(X))

J = jacobian(g(X),X); % compute Jacobian
class(J)
size(J)

%% 2.3
% Construct a CasADi function that computes the Jacobian
Jf = Function('Jf',{X},{J})

% Jf:(i0[2])->(o0[2x2]) MXFunction
%        \->            [2] means a 2-vector
%                 \->   [2x2] means a 2-by-2 matrix

%% 2.4
Jf([-0.8;2])


%% 2.5
x = [-0.8;2];

for i=1:5
    x = x - Jf(x)\g(x);
end

x

%% 3.1

% g(x) = 0
rf =  rootfinder('rf','newton',struct('x',X,'g',g(X)))

%% 3.2
rf([-0.8;2],[])

%% 3.3
rf.stats
% 6 iterations

%% 3.4
options = struct;
options.print_iteration = true;
rf =  rootfinder('rf','newton',struct('x',X,'g',g(X)),options);
rf([-0.8;2],[]);

