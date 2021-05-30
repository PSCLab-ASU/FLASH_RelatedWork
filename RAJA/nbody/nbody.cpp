//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <random>
#include <curand.h>
#include <curand_kernel.h>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

using data_type = float;

struct Particle {
  
  Particle() : pos{}, vel{}, acc{}, mass{}{};
  data_type pos[3];
  data_type vel[3];
  data_type acc[3];
  data_type mass;

};

template<typename ExecPolicy>
using INIT = RAJA::KernelPolicy<
               RAJA::statement::For<1, ExecPolicy,
                 RAJA::statement::For<0, ExecPolicy,
                    RAJA::statement::Lambda<0>
                 >
               >
             >;

using OMP_INIT = INIT<RAJA::omp_parallel_for_exec>;

using CUDA_INIT = RAJA::KernelPolicy<
                    RAJA::statement::CudaKernel<
	              RAJA::statement::For<1, RAJA::cuda_block_x_loop,
		        RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
	                  RAJA::statement::Lambda<0>
	                >
		      >
	            >
		  >;


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  
  const int N = 16000;
  const int steps = 10;
  const float tsteps = 0.1;
  const float kSofteningSquared = 1e-3f;
  const float kG = 6.67259e-11f;
  const float dt = tsteps;

  std::cout << "\nN-Body example...\n\n";


  Particle * p = memoryManager::allocate<Particle>( N );

  // _matmult_ranges_start
  RAJA::RangeSegment particle_range(0, N);
  RAJA::RangeSegment dim_range(0, 3);
  RAJA::RangeSegment zero_range(0, 0);

  // _matmult_ranges_end

  // _matmult_views_start
  RAJA::View<Particle, RAJA::Layout<1>> pview(p, N);
  // _matmult_views_end
  //RAJA::TypedIndexSet< RAJA::RangeSegment, RAJA::RangeSegment > iset;
  //iset.push_back(particle_range);
  //iset.push_back(RAJA::RangeSegment(0,3) );
  auto omp_init = [=] (int p_index, int dim) 
  {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine 
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1.0);

    pview(p_index).pos[dim] = dis(gen);
    pview(p_index).mass = N * dis(gen);
    pview(p_index).acc[dim] = 0;

    std::uniform_real_distribution<> dis2(-1, 1.0);
    pview(p_index).vel[dim] = dis2(gen);
    
  };

  auto cuda_init = [=] RAJA_DEVICE (int p_index, int dim) 
  {
     curandStatePhilox4_32_10_t state;
     curand_init(1234, 0, p_index, &state);

     pview(p_index).pos[dim] = curand_uniform(&state);
     pview(p_index).mass = N * curand_uniform(&state);
     pview(p_index).acc[dim] = 0;
  
     float vel = curand_uniform(&state) * ( 2.99999) - 1;
     pview(p_index).vel[dim] = vel;

  };

  auto omp_compute = [=] (int p_index, int )
  {
    auto& p = pview(p_index);
    float acc0 = p.acc[0];
    float acc1 = p.acc[1];
    float acc2 = p.acc[2];

    for( int j =0; j < N; j++ ){
      float dx, dy, dz;
      float distance_sqr = 0;
      float distance_inv = 0;

      dx = p.pos[0] - p.pos[0];  // 1flop
      dy = p.pos[1] - p.pos[1];  // 1flop
      dz = p.pos[2] - p.pos[2];  // 1flop

      distance_sqr = 
	      dx*dx + dy*dy + dz*dz + kSofteningSquared;
      distance_inv = 1.0f / sqrt(distance_sqr);

       acc0 += dx * kG * p.mass * distance_inv * distance_inv *distance_inv;
       acc1 += dy * kG * p.mass * distance_inv * distance_inv *distance_inv;
       acc2 += dz * kG * p.mass * distance_inv * distance_inv *distance_inv;
    }

    p.acc[0] = acc0;
    p.acc[1] = acc1;
    p.acc[2] = acc2;

  };

  auto cuda_compute = [=] RAJA_DEVICE (int p_index, int )
  {
    auto& p = pview(p_index);
    float acc0 = p.acc[0];
    float acc1 = p.acc[1];
    float acc2 = p.acc[2];

    for( int j =0; j < N; j++ ){
      float dx, dy, dz;
      float distance_sqr = 0;
      float distance_inv = 0;

      dx = p.pos[0] - p.pos[0];  // 1flop
      dy = p.pos[1] - p.pos[1];  // 1flop
      dz = p.pos[2] - p.pos[2];  // 1flop

      distance_sqr = 
	      dx*dx + dy*dy + dz*dz + kSofteningSquared;
      distance_inv = 1.0f / sqrt(distance_sqr);

       acc0 += dx * kG * p.mass * distance_inv * distance_inv *distance_inv;
       acc1 += dy * kG * p.mass * distance_inv * distance_inv *distance_inv;
       acc2 += dz * kG * p.mass * distance_inv * distance_inv *distance_inv;
    }

    p.acc[0] = acc0;
    p.acc[1] = acc1;
    p.acc[2] = acc2;

  }; 

  RAJA::ReduceSum<RAJA::omp_reduce, int>  omp_energy(0);
  RAJA::ReduceSum<RAJA::cuda_reduce, int> cuda_energy(0);

  auto omp_reduction = [=](int p_index)
  {
    auto& p = pview(p_index);
    p.vel[0] += p.acc[0] * dt;  // 2flops
    p.vel[1] += p.acc[1] * dt;  // 2flops
    p.vel[2] += p.acc[1] * dt;  // 2flops

    p.pos[0] += p.vel[0] * dt;  // 2flops
    p.pos[1] += p.vel[1] * dt;  // 2flops
    p.pos[2] += p.vel[2] * dt;  // 2flops

    p.acc[0] = 0.f;
    p.acc[1] = 0.f;
    p.acc[2] = 0.f;

    omp_energy += p.mass * p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1] + 
	                  p.vel[2] * p.vel[2];
  }; 

  auto cuda_reduction = [=] RAJA_DEVICE (int p_index)
  {
    auto& p = pview(p_index);
    p.vel[0] += p.acc[0] * dt;  // 2flops
    p.vel[1] += p.acc[1] * dt;  // 2flops
    p.vel[2] += p.acc[1] * dt;  // 2flops

    p.pos[0] += p.vel[0] * dt;  // 2flops
    p.pos[1] += p.vel[1] * dt;  // 2flops
    p.pos[2] += p.vel[2] * dt;  // 2flops

    p.acc[0] = 0.f;
    p.acc[1] = 0.f;
    p.acc[2] = 0.f;

    cuda_energy += p.mass * p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1] + 
	                   p.vel[2] * p.vel[2];

  }; 

  auto start = std::chrono::system_clock::now();
  RAJA::kernel<OMP_INIT>(RAJA::make_tuple(particle_range, dim_range),  omp_init);
  for(int i=0; i < steps; i++)
  {
    RAJA::kernel<OMP_INIT>(RAJA::make_tuple(particle_range, zero_range), omp_compute);
    RAJA::forall<RAJA::omp_parallel_for_exec>(particle_range, omp_reduction);
  }
  auto end  = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "OMP elapsed time : "<<  elapsed.count() << "us" << std::endl;

  start = std::chrono::system_clock::now();
  RAJA::kernel<CUDA_INIT>(RAJA::make_tuple(particle_range, dim_range), cuda_init);
  for(int i=0; i < steps; i++)
  {
    RAJA::kernel<CUDA_INIT>(RAJA::make_tuple(particle_range, zero_range),cuda_compute);
    RAJA::forall<RAJA::cuda_exec<256> >(particle_range, cuda_reduction);
  }
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "CUDA elapsed time : "<<  elapsed.count() << "us\n\n";
}
