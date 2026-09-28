---
title: "MonteCarloKFP.jl: Monte Carlo Solvers for Fokker–Planck Exit Problems"
description: "Notes on the mathematical scope, abstraction boundaries, GPU plans, and engineering lessons behind a Julia Monte Carlo solver for kinetic boundary-value problems."
date: 2025-03-30
lastmod: 2026-09-27
math: true
tags: ["Monte Carlo", "Fokker-Planck", "SDE", "Julia", "Scientific Computing"]
---

[MonteCarloKFP.jl](https://github.com/mingyi-ai/Monte_Carlo_KFP) began as an Apple Metal experiment and was later rebuilt as a tested Julia package. The repository README describes the current API and how to run it; this note instead records why I kept the project deliberately narrow, how its software boundaries mirror the mathematics, and what I learned while refactoring it.

## The mathematical question

The basic process is kinetic Brownian motion,

$$
\begin{aligned}
\mathrm{d}V_t &= \sigma\,\mathrm{d}W_t, \\
\mathrm{d}X_t &= V_t\,\mathrm{d}t,
\end{aligned}
$$

with generator

$$
\mathcal{L}=\frac{\sigma^2}{2}\Delta_v+v\cdot\nabla_x.
$$

Starting from a phase-space point $z=(v,x)$, trajectories are simulated until they first meet the boundary of a domain. Their exit locations approximate harmonic measure and, for boundary data $g$, the corresponding boundary-value solution

$$
u(z)=\mathbb{E}_z\!\left[g(Z_\tau)\right].
$$

What interests me most is not producing a general-purpose stochastic simulator. It is using numerical experiments to suggest how solutions behave near grazing and irregular boundary points. For a spatial boundary with outward normal $n_x$, the transport field distinguishes outgoing, incoming, and grazing points through the sign of $v\cdot n_x$. The diffusive velocity boundary and transport-outgoing boundary can carry prescribed data, while the incoming trace is determined from the interior. The transition around $v\cdot n_x=0$ is where the pictures become analytically interesting.

These computations are hints rather than proofs. They help identify singular behavior, plausible traces, and questions worth attacking analytically.

## Deliberate limits

A generic mesh domain was considered and rejected. Robust mesh geometry is a substantial and interesting computational-geometry project by itself: one must decide orientation and inside/outside semantics, classify points near faces and corners, find the first segment intersection under floating-point tolerances, handle concavity and degeneracy, and define boundary measure and sampling consistently. In higher dimensions these problems become even less suitable as a side feature of a Monte Carlo solver.

A public system of manifold patches, constructive solid geometry, or mesh composition would therefore promise much more generality than this project can responsibly support. The package instead implements a small set of analytic shapes directly. Its private line-segment and sphere routines do much of the local work that a patch implementation would do, but they remain calculation helpers rather than a public geometry language.

The code is dimension-generic, but that should not be confused with computational scalability. Kramers–Fokker–Planck models can arise in very high-dimensional settings—for example, idealized descriptions of neural-network training—but this implementation is not a practical solver for them. At one fixed starting point, ordinary Monte Carlo still has the familiar $N^{-1/2}$ sampling error, independent of dimension. The curse appears when resolving a solution over a high-dimensional domain, where the number of starting points grows exponentially, and when estimating increasingly rare exits, where the required trajectory count can also become prohibitive. This project provides neither rare-event methods nor a cure for that exponential workload.

Its realistic scope is lower-dimensional work: qualitative experiments around kinetic boundary singularities and standard two- or three-dimensional problems from statistical physics, quantum physics, and related stochastic models. That is already useful and interesting enough.

## Abstractions that follow the mathematics

The most important design choice was not a particular Julia type but the division of ownership.

### A domain owns geometric meaning

A domain determines:

- whether a point is inside, outside, or on its boundary;
- the identity and measure of each boundary component;
- how to sample those components; and
- the first boundary intersection of a path segment.

The last item belongs to the domain rather than the simulation loop. A collection of local patches does not by itself settle the global meaning of “inside,” especially for concave regions, shells, or domains defined as the exterior of absorbing targets. Keeping classification and crossing semantics together avoids forcing the trajectory code to reconstruct geometry from unrelated pieces.

Each concrete shape implements a small protocol built around a signed domain level, component levels, and a first-hit operation. Common segment and sphere calculations are shared privately. This gives the existing special shapes essentially the same internal geometric decomposition that patches would provide, without committing to a generic patch API and all of its unresolved composition rules.

### Dynamics owns the stochastic law

A dynamics object determines the noise dimension and advances one state by one Euler–Maruyama step. It receives standard-normal increments instead of drawing randomness itself. This keeps the stochastic law separate from random-number generation, thread scheduling, and hardware concerns.

The simulation layer then has a narrow job: assign a reproducible random stream to each trajectory, ask the dynamics for the next point, ask the domain whether the segment crossed its boundary, and record either the first hit or finite-horizon censoring. These responsibilities correspond closely to the mathematical objects: process, domain, stopping time, and observation.

### A boundary-value problem stays in the example

The example scripts are intentionally less abstract. Once a domain, dynamics, and stopping rule exist, a particular Dirichlet problem mostly needs a boundary function $g$ and an average of $g$ over simulated exit points. There are not yet enough distinct use cases to justify another framework around boundary data, traces, or observables.

The square example consequently contains some problem-specific and slightly rough handling of outgoing, incoming, and grazing sides. That is acceptable at the example layer: the reusable numerical and geometric semantics remain in the package, while the experiment keeps the assumptions of one boundary-value problem visible.

## Why I refactored it

The refactor came from increased software-engineering experience as much as from a numerical need. Earlier, I could ask an LLM for “engineering standards,” but I could not reliably judge the answer or steer the generated code toward a coherent design. Without understanding the desired shape of the software, more prompting did not solve the underlying problem.

Most of the current implementation was generated with AI assistance and reviewed by a human. For me, the important change is that I can now read and take responsibility for the result. Across programming languages, review starts from the user's need: what must the system mean, which component should own each rule, and where should an abstraction stop? Those boundary decisions are more important than locally polished functions.

It is tempting to explain weak generated code only as incorrect use of agents. Better prompting and better task decomposition certainly help, but they do not remove the need to read the code. My working model is that an LLM can produce many plausible, similarly rewarded implementation paths without possessing a persistent architectural intention or stable “code taste.” It can satisfy local requests while gradually mixing responsibilities or changing style. Keeping a codebase clean therefore still requires a person to select the abstractions, reject unnecessary generality, enforce consistency, and pilot the tool toward the intended whole.

The lesson was not that AI cannot write software. It was that code generation becomes useful only when the reviewer understands the demand well enough to recognize the correct boundaries.

## Metal is paused, not abandoned

The Metal backend was removed from the working tree for a practical reason: I do not currently have an available GPU device on which to develop, profile, and validate it. Maintaining accelerator code without hardware would make it nominally supported but effectively untested.

GPU execution remains on the plan. The CPU implementation now provides the reference semantics that a future Metal backend should match: the same dynamics, domain crossings, stopping behavior, censoring, and statistically consistent random sampling. When suitable hardware is available, the accelerator can return as an optional backend rather than as notebook-specific kernel code.

## Original numerical experiments

The refactor changed the organization and reliability of the code, not the mathematical experiments. The original plots are still valid and remain the most direct illustration of the project.

For the square kinetic problem, data are prescribed on the diffusive boundaries $v=\pm1$ and the transport-outgoing sides $x=-1, v<0$ and $x=1, v>0$:

![Boundary data for the square kinetic problem](/images/monte-carlo-kfp-metal/boundary_value.png)

The Monte Carlo solution displays the propagation of that data into phase space:

![Estimated solution of the square boundary-value problem](/images/monte-carlo-kfp-metal/square_boundary.png)

A closer view highlights the behavior near the singular and grazing part of the boundary:

![Zoom near the singular boundary](/images/monte-carlo-kfp-metal/solution_zoomed.png)

The same exit-sampling idea also produces a harmonic-measure distribution on an annular boundary:

![Animated annulus exit distribution](/images/monte-carlo-kfp-metal/tmp.gif)
