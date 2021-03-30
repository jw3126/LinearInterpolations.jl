module LinearInterpolations

@doc let path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    replace(read(path, String), r"^```julia"m => "```jldoctest README")
end LinearInterpolations

export interpolate, Interpolate

using ArgCheck

struct TinyVector{T} <: AbstractVector{T}
    elements::NTuple{2,T}
end
TinyVector(x, y) = TinyVector(promote(x,y))
Base.size(o::TinyVector) = (2,)
function Base.getindex(o::TinyVector, i::Integer)
    @boundscheck checkbounds(o, i)
    @inbounds ifelse(i == 1, o.elements[1], o.elements[2])
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


# struct MapProductArray{T,N,F,Factors} <: AbstractArray{T,N}
#     f::F
#     factors::Factors
# end
#
# function MapProductArray(f::F, factors::Factors) where {F,Factors}
#     N = length(factors)
#     pt = map(first, factors)
#     T = typeof(f(pt))
#     return MapProductArray{T,N,F,Factors}(f, factors)
# end
#
# function Base.getindex(o::MapProductArray{T,N}, I::Vararg{Int,N}) where {T,N}
#     @boundscheck checkbounds(o, I...)
#     xs = map(o.factors, I) do v, i
#         @inbounds v[i]
#     end
#     return o.f(xs)
# end
# Base.size(o::MapProductArray) = map(length, o.factors)

const EXTRAPOLATE_SYMBOLS = [:replicate, :reflect, :error]

function project1d(extrapolate::Symbol, xs, x)
    if extrapolate === :replicate
        project1d(Replicate(), xs, x)
    elseif extrapolate === :reflect
        project1d(Reflect(), xs, x)
    elseif extrapolate === :error
        project1d(Error(), xs, x)
    else
        @argcheck extrapolate in EXTRAPOLATE_SYMBOLS
        error("""Unreachable
              extrapolate = $extrapolate
              EXTRAPOLATE_SYMBOLS = $EXTRAPOLATE_SYMBOLS
        """)
    end
end

"""
    Replicate()

Extrapolate data by replacing out of grid points with the closest in grid point
"""
struct Replicate end

function project1d(::Replicate, xs, x)
    return clamp(x, first(xs), last(xs))
end

struct Reflect end

function project1d(::Reflect, xs, x)
    x_first = first(xs)
    x_last = last(xs)
    if x > x_last
        project1d(Reflect(), xs, 2x_last - x)
    elseif x < x_first
        project1d(Reflect(), xs, 2x_first - x)
    else
        x
    end
end

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

function _neighbors_and_weights(axes::NTuple{N,Any}, pt, extrapolate) where {N}
    nbs_wts = let extrapolate=extrapolate
        map(axes, NTuple{N}(pt)) do xs, x
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

function interpolate(axes, objs, pt; extrapolate = :error, combine::C = combine) where {C}
    itp = Interpolate(combine, axes, objs, extrapolate)
    return itp(pt)
end

function interpolate(xs::AbstractVector, ys::AbstractVector, pt; kw...)
    axes = (xs,)
    return interpolate(axes, ys, pt; kw...)
end

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
        C = typeof(combine)
        A = typeof(axes)
        V = typeof(values)
        O = typeof(extrapolate)
        return new{C,A,V,O}(combine, axes, values, extrapolate)
    end
end

function Interpolate(
    xs::AbstractVector,
    ys::AbstractVector;
    combine = combine,
    extrapolate = :error,
)
    return Interpolate(combine, (xs,), ys, extrapolate)
end
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

function (o::Interpolate)(pt)
    @argcheck length(pt) == ndims(o)
    pt2 = tupelize(o, pt)
    return o(pt2)
end

function (o::Interpolate)(pt::Tuple)
    nbs, wts = _neighbors_and_weights(o.axes, pt, o.extrapolate)
    objs = NeighborsArray(o.values, nbs)
    o.combine(wts, objs)
end

end
