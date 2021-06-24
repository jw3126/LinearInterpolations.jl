module TestCUDA
using Test
using CUDA
using Adapt
using LinearInterpolations
const LI=LinearInterpolations
CUDA.allowscalar(false)

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




end
