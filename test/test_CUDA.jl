module TestCUDA
using Test
using CUDA
using Adapt
using LinearInterpolations
using LinearInterpolations: Replicate, Reflect, WithPoint, Constant
using StructArrays
using StaticArrays
const LI=LinearInterpolations

@testset "1d" begin
    itp_cpu = Interpolate(sort!(randn(Float32, 10)), randn(Float32, 10), extrapolate=LI.Replicate())
    pts_cpu = randn(Float32, 2)

    result_cpu = itp_cpu.(pts_cpu)
    itp_gpu = adapt(CuArray, itp_cpu)
    pts_gpu = adapt(CuArray, pts_cpu)
    result_gpu = itp_gpu.(pts_gpu)

    @test result_gpu isa CuArray
    @test CuArray(result_cpu) ≈ result_gpu
end

@testset "gpu ≈ cpu" begin
    setups = Any[]
    for extrapolate in [LI.Replicate(), LI.Reflect(), LI.Constant(Float32(42)),
                        #LI.WithPoint(sin∘sum),
                       ]
        for shape_itp in [(10,), (11,12), (11,12,13)]
            axs = Tuple(sort!(randn(Float32, n)) for n in shape_itp)
            vals = randn(Float32, shape_itp...)
            itp = Interpolate(axs, vals, extrapolate=extrapolate)
            dim = length(shape_itp)
            pts = [Tuple(randn(Float32,dim)) for _ in 1:30]
            setup = (;itp, pts, extrapolate, dim)
            push!(setups, setup)
        end
    end
    @testset "$(setup.dim)D $(setup.extrapolate)" for setup in setups
        result_cpu = setup.itp.(setup.pts)
        itp_gpu = adapt(CuArray, setup.itp)
        pts_gpu = adapt(CuArray, setup.pts)
        result_gpu = itp_gpu.(pts_gpu)
        @test result_gpu isa CuArray
        @test Array(result_gpu) ≈ result_cpu

        # TODO: add soa test
        # pts_gpu_soa = StructArray(Tuple(CuArray(map(pt -> pt[i], setup.pts)) for i in 1:setup.dim))
        # pts_gpu_soa = map(SVector{setup.dim}, pts_gpu_soa)
        # @test pts_gpu_soa isa StructArray
        # result_gpu_soa = itp_gpu.(pts_gpu_soa)
        # @test result_gpu_soa isa CuArray
        # @test Array(result_gpu_soa) == Array(result_gpu)
    end
end

end#module
