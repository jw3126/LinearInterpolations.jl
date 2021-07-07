module LinearInterpolations

@doc let path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    replace(read(path, String), r"^```julia"m => "```jldoctest README")
end LinearInterpolations

export interpolate, Interpolate

using ArgCheck
import Adapt
import StaticArrays

struct TinyVector{T} <: AbstractVector{T}
    elements::NTuple{2,T}
end
TinyVector(x, y) = TinyVector(promote(x,y))
Base.size(o::TinyVector) = (2,)
Base.@propagate_inbounds function getindex(o::TinyVector, i::Integer)
    index::Int = Int(i)
    return o[index]
end
@inline function Base.getindex(o::TinyVector, i::Int)
    @boundscheck checkbounds(o, i)
    @inbounds ifelse(i === 1, o.elements[1], o.elements[2])
end

struct WeightsArray{T,N,Factors} <: AbstractArray{T,N}
    factors::Factors
end
@inline function WeightsArray(factors::NTuple{N, TinyVector}) where {N}
    T = promote_type(map(eltype, factors)...)
    Factors = typeof(factors)
    return WeightsArray{T,N,Factors}(factors)
end
function Base.size(o::WeightsArray{T,N}) where {T,N}
    return ntuple(_->2, Val(N))
end
function Base.getindex(o::WeightsArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(o, I...)
    xs = map(o.factors, I) do v, i
        @inbounds v[i]
    end
    prod(xs)::T
end

struct NeighborsArray{T,N,Objs, Inds} <: AbstractArray{T,N}
    objs::Objs
    inds::Inds
end
function Base.size(o::NeighborsArray{T,N}) where {T,N}
    ntuple(_->2, Val(N))
end
function Base.getindex(o::NeighborsArray{T,N}, I::Vararg{Int, N}) where {T,N}
    @boundscheck checkbounds(o, I...)
    inds_big = map(o.inds, I) do inds, i
        @inbounds inds[i]
    end
    @inbounds o.objs[inds_big...]
end
function NeighborsArray(objs::AbstractArray{T,N}, inds::NTuple{N,TinyVector}) where {T,N}
    Objs = typeof(objs)
    Inds = typeof(inds)
    NeighborsArray{T,N,Objs, Inds}(objs,inds)
end

const EXTRAPOLATE_SYMBOLS = [:replicate, :reflect, :error, :fuzzy]

function project1d(extrapolate::Symbol, xs, x)
    if extrapolate === :replicate
        project1d(Replicate(), xs, x)
    elseif extrapolate === :reflect
        project1d(Reflect(), xs, x)
    elseif extrapolate === :error
        project1d(Error(), xs, x)
    elseif extrapolate === :fuzzy
        project1d(Fuzzy(), xs, x)
    else
        @argcheck extrapolate in EXTRAPOLATE_SYMBOLS
        error("""Unreachable
              extrapolate = $extrapolate
              EXTRAPOLATE_SYMBOLS = $EXTRAPOLATE_SYMBOLS
        """)
    end
end

"""
    Fuzzy(;atol, rtol)

Throw an error, for data far away from the grid, but project
it onto the grid, if it is approximate accoriding to the fields
of `Fuzzy`.
"""
struct Fuzzy{K <: NamedTuple}
    kw::K
end
function Fuzzy(;kw...)
    Fuzzy((;kw...))
end

function project1d(fuzzy::Fuzzy, xs, x)
    x_inside = clamp(x, first(xs), last(xs))
    if isapprox(x, x_inside; fuzzy.kw...)
        x_inside
    else
        msg = """
        x=$x is not approximately between first(xs)=$(first(xs)) and last(xs)=$(last(xs))
        Further information:
        x = $x
        x_inside = $x_inside
        extrapolate = $fuzzy
        """
        throw(ArgumentError(msg))
    end
end

"""
    Replicate()

Extrapolate data by replacing out of grid points with the closest in grid point.
"""
struct Replicate end

function project1d(::Replicate, xs, x)
    return clamp(x, first(xs), last(xs))
end

"""
    Reflect()

Extrapolate data by reflecting out of grid points into the grid.
"""
struct Reflect end

function project1d(::Reflect, xs, x)
    x_first = first(xs)
    x_last = last(xs)
    Δx = x_last - x_first
    x1 = if x <= x_first
        n = floor((x_last-x)/(2Δx))
        x + n*2Δx
    elseif x >= x_last
        n = floor((x-x_first)/(2Δx))
        x - n*2Δx
    else
        x
    end
    x2 = if x1 > x_last
        2x_last - x1
    elseif x1 < x_first
        2x_first - x1
    else
        x1
    end
    # It should now hold that:
    # x_first <= x <= x_last
    # but floating point issues might prevent that?
    x3 = clamp(x2, x_first, x_last)
    return x3
end

"""
    Error()

Throw an error when trying to interpolate outside of the grid.
"""
struct Error end

function project1d(::Error, xs, x)
    x_first = first(xs)
    x_last = last(xs)
    if (x_first <= x <= x_last)
        x
    else
        msg = """
        x=$x is not between first(xs)=$(first(xs)) and last(xs)=$(last(xs))
        You can suppress this error by passing the `extrapolate` argument.
        """
        throw(ArgumentError(msg))
    end
end

"""
    AssumeInside()

Assume without checking, that a point is inside the grid when interpolating.
Shoud the point lie outside of the grid, behaviour is undefined.
"""
struct AssumeInside end
project1d(::AssumeInside, xs, x) = x

"""
    Constant(value)

When evaluating at a point outside the grid, return value.
"""
struct Constant{C}
    value::C
end
function apply_extrapolate(o::Constant, pt)
    o.value
end

"""
    WithPoint(f)

When evaluating at a point `pt` outside the grid, return `f(pt)`.
"""
struct WithPoint{F}
    f::F
end
function apply_extrapolate(o::WithPoint, pt)
    o.f(pt)
end

function _neighbors_and_weights1d(xs, x, extrapolate)
    x = project1d(extrapolate, xs, x)
    # searchsortedfirst: index of first value in xs greater than or equal to x
    # since we called clamp, we are inbounds
    ixu = searchsortedfirst(xs, x)
    ixu = ifelse(ixu == firstindex(xs), ixu + 1, ixu)
    ixl = ixu - 1
    xu = @inbounds xs[ixu]
    xl = @inbounds xs[ixl]
    xu_eq_xl = xu == xl
    w_tot = ifelse(xu_eq_xl, one(xu - xl), xu - xl)
    l = ifelse(xu_eq_xl, one(xu - x), xu - x)
    u = ifelse(xu_eq_xl, zero(x - xl), x - xl)
    wl = l / w_tot
    wu = u / w_tot
    wts = TinyVector(wl, wu)
    nbs = TinyVector(ixl, ixu)
    return (nbs, wts)
end

function _neighbors_and_weights(axes::NTuple{N,Any}, pt::NTuple{N,Any}, extrapolate) where {N}
    nbs_wts = let extrapolate=extrapolate
        map(axes, pt) do xs, x
            _neighbors_and_weights1d(xs, x, extrapolate)
        end
    end
    wts = WeightsArray(map(last, nbs_wts))
    nbs = map(first, nbs_wts)
    return nbs, wts
end

"""
    combine(weights::AbstractArray, objects::AbstractArray)::Object

Take a weighted collection of objects and combine them into a single object.
The default behaviour is `sum(weights .* objects)`

This `combine` can be overloaded to allow interpolation of objects that do not implement `*` or `+`.
"""
@inline function combine(wts, objs)
    _combine(wts, objs)
end
function _combine(wts::AbstractVector, objs::AbstractVector)
    w1,w2 = wts
    o1,o2 = objs
    return muladd(w2,o2,w1*o1)
end
function _combine(wts::AbstractMatrix, objs::AbstractMatrix)
    w1,w2,w3,w4 = wts
    o1,o2,o3,o4 = objs
    (w1*o1) + (w2*o2) + (w3*o3) + (w4*o4)
end
function _combine(wts::AbstractArray{Wt,3}, objs::AbstractArray{Obj,3}) where {Wt, Obj}
    w1,w2,w3,w4,w5,w6,w7,w8, = wts
    o1,o2,o3,o4,o5,o6,o7,o8 = objs
    (w1*o1) + (w2*o2) + (w3*o3) + (w4*o4) + (w5*o5) + (w6*o6) + (w7*o7) + (w8*o8)
end

function _combine(wts, objs)
    #mapreduce(*, +, wts, objs) # this allocates urgs
    w1, state_wts = iterate(wts)
    o1, state_objs = iterate(objs)
    ret = w1*o1
    while true
        next_wts = iterate(wts, state_wts)
        next_wts === nothing && break
        wi, state_wts = next_wts
        oi, state_objs = iterate(objs, state_objs)
        ret = muladd(wi,oi,ret)
    end
    return ret
end

"""
    interpolate(axes, objs, pt; extrapolate = :error, [,combine])
    interpolate(xs::AbstractVector, ys::AbstractVector, pt; kw...)

Create an `Interpolate` and evaluate it at `pt`.
See [`Interpolate`](@ref) for details.
"""
function interpolate(axes, objs, pt; extrapolate = :error, combine::C = combine) where {C}
    itp = Interpolate(combine, axes, objs, extrapolate)
    return itp(pt)
end

function interpolate(xs::AbstractVector, ys::AbstractVector, pt; kw...)
    axes = (xs,)
    return interpolate(axes, ys, pt; kw...)
end

function unsafe_Interpolate end
struct Interpolate{C,A,V,O}
    combine::C
    axes::A
    values::V
    extrapolate::O
    function Interpolate(combine, axes::Tuple, values, extrapolate)
        @argcheck size(values) == map(length, axes)
        @argcheck all(size(values) .>= 2)
        @argcheck Base.axes(values) == map(eachindex, axes)
        if extrapolate isa Symbol
            @argcheck extrapolate in EXTRAPOLATE_SYMBOLS
        end
        # some sanity checks, all of them O(ndims(values))
        # we assume issorted(xs), which wouldbe O(length(values)) to check
        @argcheck all((!isempty).(axes))
        @argcheck all(isfinite.(first.(axes)))
        @argcheck all(isfinite.(last.(axes)))
        @argcheck all(first.(axes) .<= last.(axes))
        @argcheck all(ndims.(axes) .== 1)
        @argcheck all(eachindex.(axes) .== Base.axes(values))
        LinearInterpolations.unsafe_Interpolate(combine, axes, values, extrapolate)
    end
    function LinearInterpolations.unsafe_Interpolate(combine, axes::Tuple, values, extrapolate)
        C = typeof(combine)
        A = typeof(axes)
        V = typeof(values)
        O = typeof(extrapolate)
        return new{C,A,V,O}(combine, axes, values, extrapolate)
    end
end
function Adapt.adapt_structure(to, itp::Interpolate)
    unsafe_Interpolate(
        Adapt.adapt_structure(to,itp.combine),
        Adapt.adapt_structure(to,itp.axes),
        Adapt.adapt_structure(to,itp.values),
        Adapt.adapt_structure(to,itp.extrapolate),
    )
end

function Interpolate(
    xs::AbstractVector,
    ys::AbstractVector;
    combine = combine,
    extrapolate = :error,
)
    return Interpolate(combine, (xs,), ys, extrapolate)
end

"""

    itp = Interpolate(axes, values; combine = LinearInterpolations.combine, extrapolate = :error)

Create an `Interpolate` from the given arguments. The interpolate `itp` can be evaluated on points. E.g. `itp([1,2])`.

# Arguments

* axes: The coordinates of the grid points.
* values: An AbstractArray which whose size is consistent with the length each axis.
* extrapolate: How the interpolate behaves on points outside the grid limits. Possible values are:
  - [`Replicate`](@ref), `:replicate`
  - [`Reflect`](@ref), `:reflect`
  - [`Error`](@ref), `:error`
  - [`Fuzzy`](@ref), `:fuzzy`
  - [`WithPoint`](@ref)
  - [`Constant`](@ref)
* combine: A custom function that combines weights and values into the final result.
See [`LinearInterpolations.combine`](@ref)
"""
function Interpolate(axes, values; combine = combine, extrapolate = :error)
    return Interpolate(combine, axes, values, extrapolate)
end

axestype(o::Interpolate) = axestype(typeof(o))
axestype(::Type{<:Interpolate{C,A}}) where {C,A} = A
_length(::Type{<:NTuple{N,Any}}) where {N} = N
Base.ndims(L::Type{<:Interpolate}) = _length(axestype(L))
Base.ndims(o::Interpolate) = ndims(typeof(o))

NDims(o::Interpolate) = NDims(typeof(o))
NDims(::Type{<:Interpolate{C, <: NTuple{N, Any}}}) where {C,N} = Val(N)

function _make_NTuple(itr, ::Val{0})
    return ()
end
function _make_NTuple(itr, ::Val{1})
    x1, = itr
    return (x1,)
end
function _make_NTuple(itr, ::Val{2})
    x1,x2 = itr
    promote(x1,x2)
end
function _make_NTuple(itr, ::Val{3})
    x1,x2,x3 = itr
    promote(x1,x2,x3)
end
function _make_NTuple(itr, ::Val{4})
    x1,x2,x3,x4 = itr
    promote(x1,x2,x3,x4)
end
function _make_NTuple(itr, ::Val{N}) where {N}
    ret = promote(NTuple{N,Any}(itr)...)
    ret::NTuple{N, first(ret)}
end

tupelize(itp, pt) = _make_NTuple(pt, NDims(itp))

function check_dims(itp::Itp, pt) where {Itp <: Interpolate}
    _check_dims(itp, pt, NDims(Itp))
end
_check_dims(itp, pt::NTuple{N,Any}, ::Val{N}) where {N} = nothing
_check_dims(itp, pt::Number, ::Val{1}) = nothing
_check_dims(itp, pt::StaticArrays.StaticVector{N}, ::Val{N}) where {N} = nothing
function _check_dims(itp, pt, ndims_itp)
    if ndims(itp) != length(pt)
        msg = """
        Dimensions of interpolate and point do not match. Got:
        length(pt) = $(length(pt))
        ndims(itp) = $(ndims(itp))
        """
        throw(ArgumentError(msg))
    end
end

function (itp::Interpolate)(pt)
    check_dims(itp, pt)
    pt2 = tupelize(itp, pt)
    return itp(pt2)
end

function (itp::Interpolate)(pt::Tuple)
    dispatch_extrapol(itp, pt, itp.extrapolate)
end

function eval_interpolate(itp, pt::Tuple)
    dispatch_extrapol(itp, pt, itp.extrapolate)
end

function dispatch_extrapol(itp, pt::Tuple, extrapolate)
    nbs, wts = _neighbors_and_weights(itp.axes, pt, extrapolate)
    objs = NeighborsArray(itp.values, nbs)
    itp.combine(wts, objs)
end

function isinside(pt::Tuple, axes::Tuple)
    all(
        map(pt, axes) do x, r
            first(r) <= x <= last(r)
        end
    )
end

function dispatch_extrapol(itp, pt::Tuple, extrapolate::Union{Number, AbstractArray})
    return dispatch_extrapol(itp,pt,Constant(extrapolate))
end
function dispatch_extrapol(itp, pt::Tuple, extrapolate::Union{WithPoint,
                                                                        Constant,
                                                                        })
    if isinside(pt, itp.axes)
        return dispatch_extrapol(itp, pt, AssumeInside())
    else
        T = typeof(dispatch_extrapol(itp, pt, Replicate()))
        convert(T, apply_extrapolate(extrapolate, pt))
    end
end

end
