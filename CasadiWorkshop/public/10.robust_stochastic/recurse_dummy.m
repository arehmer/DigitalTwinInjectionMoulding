function [H]=recurse_dummy(delta_num,N,history)
  % Builds a scenario tree recursively
  %
  % Settings to be passed as is: delta_num
  %
  % N: remaining horizon (shrinks while recursing)
  % history: a cell with the history of the branch (grows while recursing)
  %
  % This function return a cell with an entry for each possible series of events (=one full branch)
  % Each cell entry has a concatenation of all disturbances acting in that full branch.
  if N==0
     H = {[history{:}]};
     return  
  end
  
  H = {};
  for delta=[-delta_num,delta_num]
    H_local = recurse_dummy(delta_num,N-1,[history {delta}]);
    H = [H H_local];
  end

end