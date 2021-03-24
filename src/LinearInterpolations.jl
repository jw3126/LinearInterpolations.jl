module LinearInterpolations

export neighbors_and_weights, linterpol, Linterpol

using ArgCheck

struct MapProductArray{T,N,F,Factors} <: AbstractArray{T,N}
    f::F
    factors::Factors
end

function MapProductArray(f::F, factors::Factors) where {F, Factors}
    N = length(factors)
    pt = map(first, factors)
    T = typeof(f(pt))
    return MapProductArray{T,N,F,Factors}(f, factors)
end

function Base.getindex(o::MapProductArray{T,N}, I::Vararg{Int, N}) where {T,N}
    @boundscheck checkbounds(o, I...)
    xs = map(o.factors, I) do v, i
        @inbounds v[i]
    end
    return o.f(xs)
end

@inline function field_type(::Type{T}) where {T}
    T
end

@inline function field_type(::Type{S}, ::Type{T}) where {S, T}
    t = one(T)
    s = one(S)
    typeof((t - s) / (t + s))
end

@inline function field_type(S, T, Ts...)
    field_type(field_type(S, T), Ts...)
end

const ALLOWED_ONOUTSIDE_VALUES = [:replicate, :reflect, :error]


Base.size(o::MapProductArray) = map(length, o.factors)

function neighbors_and_weights1d_outside(xs, x, onoutside)
    if onoutside === :error
        msg = """
            x=$x is not between first(xs)=$(first(xs)) and last(xs)=$(last(xs))
            You can suppress this error by passing the `onoutside` argument.
            """
            throw(ArgumentError(msg))
    elseif onoutside === :reflect
        x_first = first(xs)
        x_last = last(xs)
        x_new = if x > x_last
            2x_last - x
        elseif x < x_first
            2x_first - x
        else
            msg = "Cannot apply onoutside=$onoutside to x=$x"
            throw(ArgumentError(msg))
        end
        return neighbors_and_weights1d(xs, x_new, onoutside)
    elseif onoutside === :replicate
        x_inside = clamp(x, first(xs), last(xs))
        return neighbors_and_weights1d(xs, x_inside, :error)
    else
        msg = """Unknown onoutside = $onoutside
        Allowed values are:
        $ALLOWED_ONOUTSIDE_VALUES
        """
        throw(ArgumentError(msg))
    end
end

struct TinyVector{T} <: AbstractVector{T}
    elements::NTuple{2,T}
    is_length_two::Bool
end

TinyVector(x)   = TinyVector((x,x), false)
TinyVector(x,y) = TinyVector((x,y), true)

function Base.size(o::TinyVector)
    (ifelse(o.is_length_two, 2, 1),)
end

function Base.getindex(o::TinyVector, i::Integer)
    @boundscheck checkbounds(o, i)
    @inbounds ifelse(i == 1, o.elements[1], o.elements[2])
end

function neighbors_and_weights1d(xs, x, onoutside=:error)
    @argcheck !isempty(xs)
    x_inside = clamp(x, first(xs), last(xs))
    is_outside = !(x ≈ x_inside)
    if is_outside
        return neighbors_and_weights1d_outside(xs, x, onoutside)
    end
    x = x_inside
    ixu = searchsortedfirst(xs,x)
    xu = xs[ixu]
    if x ≈ xu
        T   = field_type(typeof(x), eltype(xs))
        wts  = TinyVector(one(T))
        nbs = ixu:ixu
    else
        ixl = ixu - 1
        xl = xs[ixl]
        wl = (xu - x)/(xu - xl)
        wu = (x - xl)/(xu - xl)
        wts = TinyVector(wl, wu)
        nbs = ixl:ixu
    end
    return (nbs, wts)
end

function neighbors_and_weights(axes::NTuple{N, Any}, pt; onoutside=:error) where {N}
    @argcheck length(axes) == length(pt)
    nbs_wts = map(axes, NTuple{N}(pt)) do xs, x
        neighbors_and_weights1d(xs, x, onoutside)
    end
    wts = MapProductArray(prod, map(last, nbs_wts))
    nbs = map(first, nbs_wts)
    return nbs, wts
end

function combine(wts, objs)
    mapreduce(*, +, wts, objs)
end

function linterpol(axes, objs, pt; onoutside=:error, combine::C=combine) where {C}
    itp = Linterpol(combine, axes, objs, onoutside)
    return itp(pt)
end

function linterpol(xs::AbstractVector, ys::AbstractVector, pt; kw...)
    axes = (xs,)
    return linterpol(axes, ys, pt; kw...)
end

struct Linterpol{C,A,V,O}
    combine::C
    axes::A
    values::V
    onoutside::O
    function Linterpol(combine, axes, values, onoutside)
        @argcheck size(values) == map(length, axes)
        @argcheck onoutside in ALLOWED_ONOUTSIDE_VALUES
        # some sanity checks, all of them O(ndims(values))
        # we assume issorted(xs), which wouldbe O(length(values)) to check
        @argcheck all( (!isempty).(axes)                      )
        @argcheck all( isfinite.(first.(axes))                )
        @argcheck all( isfinite.(last.(axes))                 )
        @argcheck all( first.(axes) .<= last.(axes)           )
        @argcheck all( ndims.(axes) .== 1                     )
        @argcheck all( eachindex.(axes) .== Base.axes(values) )
        C = typeof(combine)
        A = typeof(axes)
        V = typeof(values)
        O = typeof(onoutside)
        return new{C,A,V,O}(combine, axes, values, onoutside)
    end
end

function Linterpol(xs::AbstractVector, ys::AbstractVector; combine=combine, onoutside=:error)
    return Linterpol(combine, (xs,), ys, onoutside)
end
function Linterpol(axes, values; combine=combine, onoutside=:error)
    return Linterpol(combine, axes, values, onoutside)
end

axestype(::Linterpol{C,A}) where {C,A} = A
axestype(::Type{<:Linterpol{C,A}}) where {C,A} = A
_length(::Type{<:NTuple{N,Any}}) where {N} = N
Base.ndims(L::Type{<:Linterpol}) = _length(axestype(L))
Base.ndims(o::Linterpol) = ndims(typeof(o))

function (o::Linterpol)(pt)
    @argcheck length(pt) == ndims(o)
    pt2 = NTuple{ndims(o)}(pt)
    return o(pt2)
end

function (o::Linterpol)(pt::Tuple)
    nbs, wts = neighbors_and_weights(o.axes, pt; onoutside=o.onoutside)
    o.combine(wts, view(o.values, nbs...))
end

end
