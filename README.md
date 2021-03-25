# LinearInterpolations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jw3126.github.io/LinearInterpolations.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jw3126.github.io/LinearInterpolations.jl/dev)
[![Build Status](https://github.com/jw3126/LinearInterpolations.jl/workflows/CI/badge.svg)](https://github.com/jw3126/LinearInterpolations.jl/actions)

# Why?
There are many excellent packages for interpolation in Julia. For instance:
* [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl)
* [Dierckx.jl](https://github.com/kbarbary/Dierckx.jl)
* [GridInterpolations.jl](https://github.com/sisl/GridInterpolations.jl)

All packages I am aware of assume, that the objects being interpolated implement addition and
scalar multiplication. However mathematically only a notion of weighted average is required for linear interpolation.
Examples of objects that support weighted average, but not addition and/or scalar multiplication are:
* Probability distributions
* Rotations and various other Lie groups

This package works with any notion of weighted average.

# Usage

```julia
julia> using LinearInterpolations

julia> xs = 1:3; ys=[10, 100, 1000]; # 1d

julia> interpolate(xs, ys, 1)
10.0

julia> interpolate(xs, ys, 1.5)
55.0

julia> pt = [1.5]; interpolate(xs, ys, pt)
55.0

julia> itp = Interpolate(xs, ys); # construct a callable for convenience

julia> itp(1.5)
55.0

julia> grid=(1:3, [10, 15]); vals = [1 2; 3 4; 5 6]; pt=[1,10]; # multi dimensional

julia> interpolate(grid, vals, pt)
1.0

julia> function winner_takes_it_all(wts, objs)
    # custom notion of weighted average
    I = argmax(wts)
    return objs[I]
end

julia> xs = 1:4; ys=[:no, :addition, :or, :multiplication];

julia> interpolate(xs, ys, 1.1, combine=winner_takes_it_all)
:no

julia> interpolate(xs, ys, 1.9, combine=winner_takes_it_all)
:addition

julia> interpolate(xs, ys, 3.7, combine=winner_takes_it_all)
:multiplication
```

# Design goals

* Lightweight and simple
* Support interpolation of objects that don't define `+,*`
* Reasonable performance
