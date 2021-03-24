using LinearInterpolations
using LinearInterpolations: neighbors_and_weights1d
using Test
using ArgCheck

@testset "neighbors_and_weights" begin
    @inferred neighbors_and_weights((1:3,), 2)
    @inferred neighbors_and_weights(([1., 2.],), Float32(2))
    @inferred neighbors_and_weights(([1., 2.],Base.OneTo(10)), [1,2])

    xs = 1:10
    x = 3.3
    axs = (xs,)
    nbs, wts = @inferred neighbors_and_weights(axs, x)
    @test nbs == (3:4,)
    @test wts ≈ [0.7, 0.3]

    axs = (1:10, 2:3)
    pt = (3.3, 2.0)
    nbs, wts = @inferred neighbors_and_weights(axs, pt)
    @test nbs == (3:4, 1:1)
    @test wts ≈ [0.7, 0.3]

end

@testset "neighbors_and_weights1d" begin
    @inferred neighbors_and_weights1d([10, 20], 30, :replicate)

    @test neighbors_and_weights1d([10, 20, 30], 10) == (1:1, [1.0])
    @test neighbors_and_weights1d([10, 20, 30], 11) == (1:2, [0.9, 0.1])
    @test neighbors_and_weights1d([10, 20, 30], 11) == (1:2, [0.9, 0.1])
    @test neighbors_and_weights1d([10, 20, 30], 20) == (2:2, [1.0])
    @test neighbors_and_weights1d([10, 20, 30], 25) == (2:3, [0.5, 0.5])
    @test neighbors_and_weights1d([10, 20, 30], 30) == (3:3, [1.0])
    @test neighbors_and_weights1d([10, 20], 30, :replicate) == (2:2, [1.0])
    @test neighbors_and_weights1d([10, 20], 0, :replicate) == (1:1, [1.0])
    @test neighbors_and_weights1d([10, 20], 25, :reflect) == (1:2, [0.5, 0.5])
    @test neighbors_and_weights1d([10, 20], 5, :reflect) == (1:2, [0.5, 0.5])
    @test neighbors_and_weights1d([10, 20], 35, :reflect) == (1:2, [0.5, 0.5])

    @test_throws ArgumentError neighbors_and_weights1d([10, 20], 30, :error)
    @test_throws ArgumentError neighbors_and_weights1d([10, 20], 30, :nonsense)
    @test_throws ArgumentError neighbors_and_weights1d([10, 20, 30], 9)
    @test_throws ArgumentError neighbors_and_weights1d([10, 20, 30], 31)
end

@testset "1d linterpol" begin
    for _ in 1:100
        xs = sort!(randn(10)); ys = randn(10);
        onoutside = rand(LinearInterpolations.ALLOWED_ONOUTSIDE_VALUES)
        if onoutside == :error
            x = clamp(randn(), xs[1], xs[end])
        else
            x = randn()
        end
        @test linterpol(xs, ys, x; onoutside=onoutside) ===
            Linterpol(xs, ys; onoutside=onoutside)(x)
    end

    @test linterpol(1:2, [10, 20], 1) ≈ 10
    @test Linterpol(1:2, [10, 20])(1) ≈ 10
    @test linterpol(1:2, [10, 20], 1.5) ≈ 15
    @test linterpol(1:2, [10, 20], 2) ≈ 20
    @test linterpol(1:2, [10, 20], 2f0) === 20f0
    @test linterpol(1:2, [10, 20], 2.0) === 20.0
    for onoutside in [:error, :reflect, :replicate]
        @test linterpol(1:2, [10, 20], 2.0; onoutside) === 20.0
    end
    @test linterpol([10, 20], [2, 1], 30000, onoutside=:replicate) ≈ 1
    @test linterpol([10, 20], [2, 1], 30, onoutside=:reflect) ≈ 2
    @test linterpol([10, 20], [2, 1], 30, onoutside=:reflect) ≈ 2
    @test linterpol([10, 20], [2, 1], 22.5, onoutside=:reflect) ≈ 1.25

    @inferred linterpol(1:2, [10, 20], 1)
    @inferred linterpol(1:2, [10, 20], 1f0)
    @inferred linterpol(1:2, [10, 20], 1.0)
    @inferred linterpol(1:2, [10, 20], 1.0, onoutside=:error)
    @test_throws ArgumentError linterpol(1:2, [10, 20, 30], 0.9)
    @test_throws ArgumentError linterpol(1:2, [10, 20, 30], 2.1)
    @test_throws ArgumentError linterpol(1:2, [10, 20], 2.1, onoutside=:nonsense)
    @test_throws ArgumentError Linterpol(1:2, [10, 20], onoutside=:nonsense)

    @testset "double points" begin
        @test linterpol([1,1,2], [10,20,30], 1.5) ≈ 25
        @test linterpol([1,1,2], [10,20,30], 1.000000001) ≈ 20
        @test 10 <= linterpol([1,1,2], [10,20,30], 1.0) <= 20

        @test linterpol([0,1,1,2], [0,10,20,30], 1.5) ≈ 25
        @test linterpol([0,1,1,2], [0,10,20,30], 1.000000001) ≈ 20
        @test 10 <= linterpol([0,1,1,2], [0,10,20,30], 1.0) <= 20
    end
end

@testset "Probability distributions" begin
    struct ProbDist
        probabilities::Vector{Float64}
        function ProbDist(ps)
            @argcheck sum(ps) ≈ 1
            @argcheck all(p -> p>=0, ps)
            new(Vector{Float64}(ps))
        end
    end

    function LinearInterpolations.combine(wts, dists::AbstractArray{ProbDist})
        ps = LinearInterpolations.combine(wts, map(dist-> dist.probabilities, dists))
        ProbDist(ps)
    end

    wts  = [0.1, 0.9]
    dists = [ProbDist([0.5, 0.5]), ProbDist([1, 0])]
    dist = LinearInterpolations.combine(wts, dists)
    @test dist isa ProbDist
    @test dist.probabilities ≈ [0.95, 0.05]

    xs = [10, 20]
    pt = 19
    @test linterpol(xs, dists, pt).probabilities ≈ [0.95, 0.05]
    @test Linterpol(xs, dists)(pt).probabilities ≈ [0.95, 0.05]

    @inferred linterpol(xs, dists, pt)
    @inferred Linterpol(xs, dists)(pt)

end
