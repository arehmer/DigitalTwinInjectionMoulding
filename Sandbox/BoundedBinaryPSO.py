# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:33:48 2021

@author: LocalAdmin
"""

# Import standard library
import logging

# Import modules
import numpy as np
import multiprocessing as mp

from collections import deque


from pyswarms.discrete.binary import BinaryPSO
from pyswarms.backend.operators import compute_pbest, compute_objective_function
from pyswarms.backend.topology import Ring
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.base import DiscreteSwarmOptimizer
from pyswarms.utils.reporter import Reporter


class BoundedBinaryPSO(BinaryPSO):
    def __init__(
        self,
        n_particles,
        dimensions_discrete,
        options,
        bounds,
        bh_strategy="periodic",
        init_pos=None,
        velocity_clamp=None,
        vh_strategy="unmodified",
        ftol=-np.inf,
        ftol_iter=1,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                * k : int
                    number of neighbors to be considered. Must be a
                    positive integer less than :code:`n_particles`
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the
                    sum-of-absolute values (or L1 distance) while 2 is
                    the Euclidean (or L2) distance.
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh_strategy : String
            a strategy for the handling of the velocity of out-of-bounds particles.
            Only the "unmodified" and the "adjust" strategies are allowed.
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        """
        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options["k"], options["p"]
        
        self.dimensions_discrete = dimensions_discrete
        
        self.bits,self.bounds = self.translate_discrete_to_binary(
            dimensions_discrete,bounds)
        
        
        # Initialize parent class
        super(BinaryPSO, self).__init__(
            n_particles=n_particles,
            dimensions=sum(self.bits),
            binary=True,
            options=options,
            init_pos=init_pos,
            velocity_clamp=velocity_clamp,
            ftol=ftol,
            ftol_iter=ftol_iter,
        )
        # self.bounds = bounds
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Ring(static=False)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.name = __name__
        
        
        
    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, **kwargs
        ):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int, optional
            number of processes to use for parallel particle evaluation
            Defaut is None with no parallelization.
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for objective function

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            self.swarm.current_cost = compute_objective_function(
                self.swarm, objective_func, pool, **kwargs
            )
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm
            )
            best_cost_yet_found = np.min(self.swarm.best_cost)
            # Update gbest from neighborhood
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, p=self.p, k=self.k
            )
            if verbose:
                # Print to console
                self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=np.mean(self.swarm.best_cost),
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform position velocity update
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh
            )
            self.swarm.position = self._compute_position(self.swarm)
            
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()

        return (final_best_cost, final_best_pos)
    
    # def _compute_position(
    #     self, swarm, bounds=None, bh=BoundaryHandler(strategy="periodic")
    # ):
    #     """Update the position matrix of the swarm

    #     This computes the next position in a binary swarm. It compares the
    #     sigmoid output of the velocity-matrix and compares it with a randomly
    #     generated matrix.

    #     Parameters
    #     ----------
    #     swarm: pyswarms.backend.swarms.Swarm
    #         a Swarm class
    #     """
        
        
    #     temp_position = (np.random.random_sample(size=swarm.dimensions)
    #         < self._sigmoid(swarm.velocity)) * 1
        
    #     #  Not necessary, bounds are included in binary coding !!!
    #     # if bounds is not None:
    #     #     # Calculate binary positions back to real positions
    #     #     temp_position_real = self.BinarySwarm_to_DiscreteSwarm(temp_position)
    #     #     temp_position_real = bh(temp_position_real, bounds)
    #     #     # Calculate bounded real positions back to binary positions
    #     #     temp_position = f(temp_position_real)
    #     # position = temp_position
        
    #     # print(r)
        
    #     return temp_position
    
    def translate_discrete_to_binary(self,dimensions,bounds):
        """
        Calculates the number of bits necessary to represent a discrete
        optimization problem with "dimensions" number of discrete variables
        as a binary optimization problem with a certain number of bits and
        adjusts bounds accordingly
        
        Parameters
        ----------  
        dimensions: integer, number of discrete variables
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        """
        
        bits = []
        
        for n in range(0,dimensions):
            
            # Number of bits required rounding down!
            bits.append(int(np.log10(bounds[1][n]-bounds[0][n]+1) / np.log10(2)))
        
            # Adjust upper bound accordingly
            bounds[1][n] = bounds[0][n] + 2**bits[n]-1
        
        return bits, bounds

    def BinarySwarm_to_DiscreteSwarm(self,binary_position):
        
        discrete_position = np.zeros((self.n_particles,self.dimensions_discrete))
        
        cum_sum = 0
        
        for i in range(0,len(self.bits)):
            
            bit = self.bits[i]
            lb = self.bounds[0][i]
            
            discrete_position[:,cum_sum:bit] = lb + \
            self.bool2int(binary_position[:,cum_sum:cum_sum+bit])
            
            cum_sum = cum_sum + bit
        return discrete_position    
                    
    def bool2int(self,x):
        
        x_int = np.zeros((x.shape[0],1))
        
        for row in range(0,x.shape[0]):
            row_int = 0
            
            for i,j in enumerate(x[row,:]):
                row_int += j<<i
            
            x_int[row] = row_int
        
        return x_int          
        
    def float_to_binary(x, m, n):
        """Convert the float value `x` to a binary string of length `m + n`
        where the first `m` binary digits are the integer part and the last
        'n' binary digits are the fractional part of `x`.
        """
        x_scaled = round(x * 2 ** n)
        return '{:0{}b}'.format(x_scaled, m + n)
    
    def binary_to_float(bstr, m, n):
        """Convert a binary string in the format given above to its float
        value.
        """
        return int(bstr, 2) / 2 ** n        
        
     
        
        
        
        
        