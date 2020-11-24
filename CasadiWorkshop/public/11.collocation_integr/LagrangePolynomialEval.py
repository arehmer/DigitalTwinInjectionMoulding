import numpy as np

def LagrangePolynomialEval(X,Y,x):
    if isinstance(X,list):
      X = np.array(X)
    if isinstance(Y,list):
      Y = np.array(Y)
    assert not isinstance(X,list)
    # Interpolates exactly through data 
    if len(Y.shape)>1:
      assert(1 in X.shape)
    N = np.prod(X.shape)
    if len(Y.shape)==1:
      Y = Y.reshape((1,N))
    else:
      assert(Y.shape[1]==N)

    res = 0

    for j in range(N):
        p = 1
        for i in range(N):
            if i!=j:
               p = p*(x-X[i])/(X[j]-X[i])

        res = res+p*Y[:,j]
    return res


