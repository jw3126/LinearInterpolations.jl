using LinearInterpolations
using LinearInterpolations: neighbors_and_weights1d, neighbors_and_weights
using Test
using ArgCheck

@testset "neighbors_and_weights" begin
    @inferred neighbors_and_weights((1:3,), 2, :error)
    @inferred neighbors_and_weights(([1.0, 2.0],), Float32(2), :error)
    @inferred neighbors_and_weights(([1.0, 2.0], Base.OneTo(10)), [1, 2], :error)

    xs = 1:10
    x = 3.3
    axs = (xs,)
    nbs, wts = @inferred neighbors_and_weights(axs, x, :error)
    @test nbs == (3:4,)
    @test wts ≈ [0.7, 0.3]

    axs = (1:10, 2:3)
    pt = (3.3, 2.0)
    nbs, wts = @inferred neighbors_and_weights(axs, pt, :error)
    @test nbs == (3:4, 1:1)
    @test wts ≈ [0.7, 0.3]

end

@testset "neighbors_and_weights1d" begin
    @inferred neighbors_and_weights1d([10, 20], 30, :replicate)

    @test neighbors_and_weights1d([10, 20, 30], 10, :error) == (1:1, [1.0])
    @test neighbors_and_weights1d([10, 20, 30], 11, :error) == (1:2, [0.9, 0.1])
    @test neighbors_and_weights1d([10, 20, 30], 11, :error) == (1:2, [0.9, 0.1])
    @test neighbors_and_weights1d([10, 20, 30], 20, :error) == (2:2, [1.0])
    @test neighbors_and_weights1d([10, 20, 30], 25, :error) == (2:3, [0.5, 0.5])
    @test neighbors_and_weights1d([10, 20, 30], 30, :error) == (3:3, [1.0])
    @test neighbors_and_weights1d([10, 20], 30, :replicate) == (2:2, [1.0])
    @test neighbors_and_weights1d([10, 20], 0, :replicate) == (1:1, [1.0])
    @test neighbors_and_weights1d([10, 20], 25, :reflect) == (1:2, [0.5, 0.5])
    @test neighbors_and_weights1d([10, 20], 5, :reflect) == (1:2, [0.5, 0.5])
    @test neighbors_and_weights1d([10, 20], 35, :reflect) == (1:2, [0.5, 0.5])

    @test_throws ArgumentError neighbors_and_weights1d([10, 20], 30, :error)
    @test_throws ArgumentError neighbors_and_weights1d([10, 20], 15, :nonsense)
    @test_throws ArgumentError neighbors_and_weights1d([10, 20], 30, :nonsense)
    @test_throws ArgumentError neighbors_and_weights1d([10, 20, 30], 9,:error)
    @test_throws ArgumentError neighbors_and_weights1d([10, 20, 30], 31, :error)
end

@testset "1d interpolate" begin
    for _ = 1:100
        xs = sort!(randn(10))
        ys = randn(10)
        extrapolate = rand(LinearInterpolations.EXTRAPOLATE_SYMBOLS)
        if extrapolate == :error
            x = clamp(randn(), xs[1], xs[end])
        else
            x = randn()
        end
        @inferred interpolate(xs, ys, x; extrapolate = extrapolate)
        @inferred Interpolate(xs, ys; extrapolate = extrapolate)
        @test interpolate(xs, ys, x; extrapolate = extrapolate) ===
              Interpolate(xs, ys; extrapolate = extrapolate)(x)
    end

    @test interpolate(1:2, [10, 20], 1) ≈ 10
    @test Interpolate(1:2, [10, 20])(1) ≈ 10
    @test interpolate(1:2, [10, 20], 1.5) ≈ 15
    @test interpolate(1:2, [10, 20], 2) ≈ 20
    @test interpolate(1:2, [10, 20], 2.0f0) === 20.0f0
    @test interpolate(1:2, [10, 20], 2.0) === 20.0
    for extrapolate in [:error, :reflect, :replicate]
        @test interpolate(1:2, [10, 20], 2.0; extrapolate) === 20.0
    end
    @test interpolate([10, 20], [2, 1], 30000, extrapolate = :replicate) ≈ 1
    @test interpolate([10, 20], [2, 1], 30, extrapolate = :reflect) ≈ 2
    @test interpolate([10, 20], [2, 1], 22.5, extrapolate = :reflect) ≈ 1.25

    @inferred interpolate(1:2, [10, 20], 1)
    @inferred interpolate(1:2, [10, 20], 1.0f0)
    @inferred interpolate(1:2, [10, 20], 1.0)
    @inferred interpolate(1:2, [10, 20], 1.0, extrapolate = :error)
    @test_throws ArgumentError interpolate(1:2, [10, 20, 30], 0.9)
    @test_throws ArgumentError interpolate(1:2, [10, 20, 30], 2.1)
    @test_throws ArgumentError interpolate(1:2, [10, 20], 2.1, extrapolate = :nonsense)
    @test_throws ArgumentError Interpolate(1:2, [10, 20], extrapolate = :nonsense)

    @testset "double points" begin
        @test interpolate([1, 1, 2], [10, 20, 30], 1.5) ≈ 25
        @test interpolate([1, 1, 2], [10, 20, 30], 1.000000001) ≈ 20
        @test 10 <= interpolate([1, 1, 2], [10, 20, 30], 1.0) <= 20

        @test interpolate([0, 1, 1, 2], [0, 10, 20, 30], 1.5) ≈ 25
        @test interpolate([0, 1, 1, 2], [0, 10, 20, 30], 1.000000001) ≈ 20
        @test 10 <= interpolate([0, 1, 1, 2], [0, 10, 20, 30], 1.0) <= 20
    end
end

@testset "interpolate 2d" begin
    grid = (1:2, 1:3)
    vals = [(1, 1) (1, 2) (1, 3); (2, 1) (2, 2) (2, 3)]
    pt = [1.5, 1.1]
    wts, nbs = interpolate(grid, vals, pt, combine = tuple)
    @test nbs == [(1, 1) (1, 2); (2, 1) (2, 2)]
    @test wts ≈ [0.45 0.05; 0.45 0.05]

    grid = (1:2, 1:3)
    vals = randn(2, 3)
    pt = [1.5, 1.1]
    ret = interpolate(grid, vals, pt)
    @test ret ≈
          0.45 * vals[1, 1] + 0.45 * vals[2, 1] + 0.05 * vals[1, 2] + 0.05 * vals[2, 2]
end

@testset "interpolate 3d" begin
    for _ = 1:10
        grid = (1:2, 1:3, 4:7)
        vals = randn(2, 3, 4)
        pt = [1.5, 1.1, 6.2]
        # ret = interpolate(grid, vals, pt)
        ret = interpolate(grid, vals, pt)
        wl1 = 0.5
        wu1 = 0.5
        wl2 = 0.9
        wu2 = 0.1
        wl3 = 0.8
        wu3 = 0.2
        il1 = 1
        iu1 = 2
        il2 = 1
        iu2 = 2
        il3 = 3
        iu3 = 4
        @test ret ≈
              wl1 * wl2 * wl3 * vals[il1, il2, il3] +
              wl1 * wl2 * wu3 * vals[il1, il2, iu3] +
              wl1 * wu2 * wl3 * vals[il1, iu2, il3] +
              wl1 * wu2 * wu3 * vals[il1, iu2, iu3] +
              wu1 * wl2 * wl3 * vals[iu1, il2, il3] +
              wu1 * wl2 * wu3 * vals[iu1, il2, iu3] +
              wu1 * wu2 * wl3 * vals[iu1, iu2, il3] +
              wu1 * wu2 * wu3 * vals[iu1, iu2, iu3]
    end
end

@testset "Probability distributions" begin
    struct ProbDist
        probabilities::Vector{Float64}
        function ProbDist(ps)
            @argcheck sum(ps) ≈ 1
            @argcheck all(p -> p >= 0, ps)
            new(Vector{Float64}(ps))
        end
    end

    function LinearInterpolations.combine(wts, dists::AbstractArray{ProbDist})
        ps = LinearInterpolations.combine(wts, map(dist -> dist.probabilities, dists))
        ProbDist(ps)
    end

    wts = [0.1, 0.9]
    dists = [ProbDist([0.5, 0.5]), ProbDist([1, 0])]
    dist = LinearInterpolations.combine(wts, dists)
    @test dist isa ProbDist
    @test dist.probabilities ≈ [0.95, 0.05]

    xs = [10, 20]
    pt = 19
    @test interpolate(xs, dists, pt).probabilities ≈ [0.95, 0.05]
    @test Interpolate(xs, dists)(pt).probabilities ≈ [0.95, 0.05]

    @inferred interpolate(xs, dists, pt)
    @inferred Interpolate(xs, dists)(pt)
end

@testset "internals" begin


    xs = [-10.0, 0.0, 10.0]
    itp = @inferred Interpolate((xs,xs), randn(3,3))
    pt = (1.0, 2.0)
    @show @allocated itp(pt)
    @show @allocated itp(pt)
    @inferred LinearInterpolations.tupelize(itp, [1.0, 2.0])
    @inferred LinearInterpolations.tupelize(itp, [1, 2])
    @inferred LinearInterpolations.tupelize(itp, (1.0, 2))

    itp = @inferred Interpolate((xs,xs,xs), randn(3,3,3))
    @inferred LinearInterpolations.tupelize(itp, [1  , 2, 1.0])
    @inferred LinearInterpolations.tupelize(itp, [1  , 2, 3  ])
    @inferred LinearInterpolations.tupelize(itp, (1.0, 2, 1f0))

    @inferred interpolate((xs,xs,xs,xs), randn(3,3,3,3), [1.9,2,3,4])
    itp = @inferred Interpolate((xs,xs,xs,xs), randn(3,3,3,3))
    @inferred LinearInterpolations.tupelize(itp, [1  , 2, 1.0, 2.0 ])
    @inferred LinearInterpolations.tupelize(itp, [1  , 2, 3  , 4   ])
    @inferred LinearInterpolations.tupelize(itp, (1.0, 2, 1f0, 0x12))

    @testset "no allocs $(dim)d" for dim in 1:4
        xs = [-10,0,20.0]
        axs = ntuple(_-> xs, dim)
        vals = randn(ntuple(_->3, dim)...)
        itp = @inferred Interpolate(axs, vals, extrapolate=LinearInterpolations.Replicate())
        pt = ntuple(_->0.0, dim)
        @inferred itp(pt)
        @test_broken 0 == @allocated itp(pt)
    end
end
