def recurse_dummy(delta_num,N,history):
  # Builds a scenario tree recursively
  #
  # Settings to be passed as is: delta_num
  #
  # N: remaining horizon (shrinks while recursing)
  # history: a list with the history of the current full branch (grows while recursing)

  # This function return a cell with an entry for each possible series of events (=one full branch)
  # Each cell entry has a concatenation of all disturbances acting in that full branch.

  from casadi import vertcat, hcat
  if N==0:
     return [hcat(history)]
  
  H = []
  for delta in [-delta_num,delta_num]:
    H_local = recurse_dummy(delta_num,N-1,history+[delta])
    H+=H_local
  return H
