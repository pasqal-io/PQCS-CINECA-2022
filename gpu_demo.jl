```
To run this script:
- install julia 1.8
- check out https://github.com/ITensor/ITensors.jl.git at commit 6f6831e20ef5badc5ac82350e7524c968c876b4f
- julia -e 'using Pkg; Pkg.develop(path="<ITENSOR_ROOT>")'
- julia -e 'using Pkg; Pkg.develop(path="<ITENSOR_ROOT>/ITensorGPU")'
```
using ITensors
using CUDA
using CUDA.CUTENSOR

N = 100  

function state_factors(raw_states, sites, links)
  state_factors::Vector{ITensor} = [ITensor(raw_states[1], sites[1], links[1])]
  for i in 2:N-1
    factor = ITensor(raw_states[i], links[i-1], sites[i], links[i])
    push!(state_factors, factor)
  end
  push!(state_factors, ITensor(raw_states[N], links[N-1], sites[N]))
  return state_factors
end

function rotation_layer(raw_rotations, sites)
    ops::Vector{ITensor}=[]
    for (i,s) in enumerate(sites)
      push!(ops, ITensor(raw_rotations[i], s', s))
    end
    return ops
end

function cu_state_factors(raw_states, cusites, culinks)
  cu_state_factors::Vector{CuTensor} = [CuTensor(ComplexF64.(cu(raw_states[1])), [cusites[1], culinks[1]])]
  for i in 2:N-1
    factor = CuTensor(ComplexF64.(cu(raw_states[i])), [culinks[i-1], cusites[i], culinks[i]])
    push!(cu_state_factors, factor)
  end
  push!(cu_state_factors, CuTensor(ComplexF64.(cu(raw_states[N])), [culinks[N-1], cusites[N]]))
  return cu_state_factors
end

function cu_rotation_layer(raw_rotations, sites, sitesprime)
  ops::Vector{CuTensor}=[]
  for i in 1:N
    push!(ops, CuTensor(ComplexF64.(cu(raw_rotations[i])), [sitesprime[i], sites[i]]))
  end
  return ops
end

function time_cpu_and_cuda(bond_dim)
  angles = [pi/2 for _ in 1:3N]

  Ry(theta) = cos(theta/2)*[1 0; 0 1] + im*sin(theta/2)*[0 -im; im 0]
  Rx(theta) = cos(theta/2)*[1 0; 0 1] + im*sin(theta/2)*[0 1; 1 0]
  rotations(t1::Number, t2::Number, t3::Number) = Ry(t3)*Rx(t2)*Ry(t1)

  raw_states::Vector{AbstractArray} = [rand(2, bond_dim)+im*rand(2, bond_dim)]
  for i in 2:N-1
    push!(raw_states, rand(bond_dim, 2, bond_dim)+im*rand(bond_dim, 2, bond_dim))
  end
  push!(raw_states, rand(bond_dim, 2)+im*rand(bond_dim, 2))

  raw_rotations = []
  for i in 1:N
    push!(raw_rotations, rotations(angles[1+3*(i-1)], angles[2+3*(i-1)], angles[3+3*(i-1)]))
  end

  sites = [Index(2) for _ in 1:N]
  links = [Index(bond_dim) for _ in 1:N-1]

  in_factors = state_factors(raw_states, sites, links)
  layer = rotation_layer(raw_rotations, sites)
  t1 = @timed contract.(in_factors, layer)


  cusites = [Char(i) for i in 1:N]
  cusitesprime = [Char(i) for i in N+1:2N]
  culinks = [Char(i) for i in 2N+1:3N]

  cu_in_factors = cu_state_factors(raw_states, cusites, culinks)
  cu_layer = cu_rotation_layer(raw_rotations, cusites, cusitesprime)
  t2 = CUDA.@elapsed cu_in_factors .* cu_layer

  println("For bond dim $bond_dim, CPU took $(t1[:time])")
  println("For bond dim $bond_dim, GPU took $t2")
  println("-----------------")
  return (CPU=t1[:time], GPU=t2)
end

time_cpu_and_cuda(10)
time_cpu_and_cuda(20)
time_cpu_and_cuda(40)
time_cpu_and_cuda(80)
time_cpu_and_cuda(160)
time_cpu_and_cuda(320)
time_cpu_and_cuda(640)