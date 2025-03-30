---
title: "Monte Carlo Simulation for Fokker-Planck Equations using Metal"
tags: ["Monte Carlo", "Fokker-Planck", "Metal", "GPU", "Julia"]
---

This project implements a Monte Carlo simulation for the Fokker-Planck equation using `Metal`, a framework for GPU programming on Apple devices. The simulation is designed to model the diffusion of particles in a potential field, which is common in statistical physics and machine learning.

The codes are written in `Julia` and utilize the `Metal.jl` package to leverage the GPU for parallel computation. You can find the repository for this project [here](https://github.com/mingyi-ai/Monte_Carlo_KFP).

## Kramers-Fokker-Planck Equation

The Kramers-Fokker-Planck equation is the generator of stochastic process modeling a particle's movement in a potential field.
The stationary solution of the Fokker-Planck equation is the equilibrium distribution of the particle's position.

One example considered in this project is the following SDE:
$$
\begin{cases}
    \rm{d}X_t = V_t \rm{d}t,\newline
    \rm{d}V_t = \rm{d}B_t,
\end{cases}
$$
where $B_t$ is the standard Brownian motion.
Denote by $z = (x, v)\in \mathbb{R}\times\mathbb{R}$.
Its generator is given by:
$$\cal{L} = \frac{1}{2}\frac{\partial^2}{\partial v^2} + v\frac{\partial}{\partial x}.$$

We consider the process in the box $Q = (-1, 1)\times(-1, 1)$, and simulate the process starting at a point $z_0 = (x_0, v_0)\in Q$ until it hits the boundary of the box.

The Monte Carlo simulation involves:

- A Metal kernel function that updates the position of the particle according to the discretized SDE, namely $$ x_{n+1} = x_n + v_n \Delta t, \quad v_{n+1} = v_n + \sqrt{\Delta t} \xi_n,$$
where $\xi_n \sim \mathcal{N}(0, 1)$ is the standard normal random variable.
- Since the Metal kernel does not support random number generation, we also write a simple GPU friendly random number generator based on xorshift32 and Box-Muller method.
- The Kernel function does not support branching, the iteration will be fixed steps, and we use mask to stop the iteration when the particle hits the boundary.
- We simulate the process for a large number of particles and plot the harmonic measure on the boundary of the annulus.

## Features of the Project

### GPU friendly random number generator

The random number generator is based on the xorshift32 algorithm, which is a simple and efficient pseudo-random number generator. The Box-Muller transform is used to generate normally distributed random numbers from uniformly distributed random numbers.

Given a seed ranged from 0 to 2^32-1, the xorshift32 algorithm generates a new seed by performing bitwise operations on the current seed.

```Julia
function xorshift32(seed::UInt32)::UInt32
    seed ⊻= (seed << 13)
    seed ⊻= (seed >> 17)
    seed ⊻= (seed << 5)
    return seed
end
```

Then we transform this seed to a float number in the range of (0, 1) using the following function:

```Julia
function xorshift32_float(seed::UInt32)::Float32
    value = Float32(xorshift32(seed)) * 2.3283064f-10  # Scale to [0,1)
    return max(value, 1.0f-16)  # Ensure it's in (0,1)
end
```

Finally, we use the Box-Muller transform to generate normally distributed random numbers:

```Julia
function box_muller(u1::Float32, u2::Float32)
    r = sqrt(-2.0f0 * log(u1))
    theta = 2.0f0 * Float32(pi) * u2
    return r * cos(theta)
end
```

### Masks to avoid branching

The `Metal.jl` kernel does not support branching, so we need to avoid using `if` statements in the kernel code. Instead, we use masks to control the flow of the simulation. The core update function for the problem in the cube $Q$ is as follows:

```Julia
for step in 1:num_steps
        # Boolean masks for exit conditions
        mask_x = (x < -1.0f0 || x > 1.0f0) ? 1 : 0
        mask_v = (v < -1.0f0 || v > 1.0f0) ? 1 : 0
        mask_exit = mask_x | mask_v  # Combine masks (exit if either condition is true)
        continue_mask = 1 - mask_exit  # 1 = active, 0 = exited

        # Generate two uniform distributed random numbers
        seed1 = xorshift32(seed1)
        seed2 = xorshift32(seed2)
        random_number1 = xorshift32_float(seed1)
        random_number2 = xorshift32_float(seed2)

        # Generate a normal distributed noise
        noise = box_muller(random_number1, random_number2)

        # Perturb the seeds to avoid deterministic patterns
        seed1 += UInt32(i)
        seed2 += UInt32(i)

        # Update position and velocity and store previous state if not exit
        x_prev, v_prev = continue_mask * x + mask_exit * x_prev, continue_mask * v + mask_exit * v_prev 
        x += continue_mask * (v * time_step)
        v += continue_mask * (sqrt(time_step) * noise)
    end
```

The `mask_exit` variable is used to check if the particle has exited the box. If it has, we set the `continue_mask` to 0, which effectively stops the simulation for that particle. The `x_prev` and `v_prev` variables are used to store the previous state of the particle before it exited.

### Example plots

Consider the following Dirichlet boundary condition:
![Boundary Value Plot](/images/monte-carlo-kfp-metal/boundary_value.png)
Our codes can simulate the solution efficiently. The following plot shows the full solution and also a zoomed-in view of the solution near the singular boundary:
![Full Solution Plot](/images/monte-carlo-kfp-metal/square_boundary.png)
![Zoomed Solution Plot](/images/monte-carlo-kfp-metal/solution_zoomed.png)

In addition, we can plot the exit points distribution on the boundary for a starting point. The following is an example in the annulus:
![Hitting Distribution Plot](/images/monte-carlo-kfp-metal/tmp.gif)