module LinearInterpolations

@doc let path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    replace(read(path, String), r"^```julia"m => "```jldoctest README")
end LinearInterpolations

export neighbors_and_weights, interpolate, Interpolate

using ArgCheck

struct MapProductArray{T,N,F,Factors} <: AbstractArray{T,N}
    f::F
    factors::Factors
end

function MapProductArray(f::F, factors::Factors) where {F,Factors}
    N = length(factors)
    pt = map(first, factors)
    T = typeof(f(pt))
    return MapProductArray{T,N,F,Factors}(f, factors)
end

function Base.getindex(o::MapProductArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(o, I...)
    xs = map(o.factors, I) do v, i
        @inbounds v[i]
    end
    return o.f(xs)
end

@inline function field_type(::Type{T}) where {T}
    T
end

@inline function field_type(::Type{S}, ::Type{T}) where {S,T}
    t = one(T)
    s = one(S)
    typeof((t - s) / (t + s))
end

@inline function field_type(S, T, Ts...)
    field_type(field_type(S, T), Ts...)
end


Base.size(o::MapProductArray) = map(length, o.factors)

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

struct TinyVector{T} <: AbstractVector{T}
    elements::NTuple{2,T}
    is_length_two::Bool
end

TinyVector(x) = TinyVector((x, x), false)
TinyVector(x, y) = TinyVector((x, y), true)

function Base.size(o::TinyVector)
    (ifelse(o.is_length_two, 2, 1),)
end

function Base.getindex(o::TinyVector, i::Integer)
    @boundscheck checkbounds(o, i)
    @inbounds ifelse(i == 1, o.elements[1], o.elements[2])
end

function neighbors_and_weights1d(xs, x, extrapolate = :error)
    x = project1d(extrapolate, xs, x)
    # searchsortedfirst: index of first value in xs greater than or equal to x
    # since we called clamp, we are inbounds
    ixu = searchsortedfirst(xs, x)
    xu = @inbounds xs[ixu]
    if x == xu
        T = field_type(typeof(x), eltype(xs))
        wts = TinyVector(one(T))
        nbs = ixu:ixu
    else
        ixl = ixu - 1
        xl = @inbounds xs[ixl]
        wl = (xu - x) / (xu - xl)
        wu = (x - xl) / (xu - xl)
        wts = TinyVector(wl, wu)
        nbs = ixl:ixu
    end
    return (nbs, wts)
end

function neighbors_and_weights(axes::NTuple{N,Any}, pt; extrapolate = :error) where {N}
    @argcheck length(axes) == length(pt)
    nbs_wts = map(axes, NTuple{N}(pt)) do xs, x
        neighbors_and_weights1d(xs, x, extrapolate)
    end
    wts = MapProductArray(prod, map(last, nbs_wts))
    nbs = map(first, nbs_wts)
    return nbs, wts
end

"""
    combine(weights::AbstractArray, objects::AbstractArray)::Object

Take a weighted collection of objects and combine them into a single object.
The default behaviour is `sum(weights .* objects)`

This `combine` can be overloaded to allow interpolation of objects that do not implement `*` or `+`.
"""
function combine(wts, objs)
    mapreduce(*, +, wts, objs)
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
    nbs, wts = neighbors_and_weights(o.axes, pt; extrapolate = o.extrapolate)
    o.combine(wts, view(o.values, nbs...))
end

end
