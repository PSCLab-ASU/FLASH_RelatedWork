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
  
  Particle() : pos{} {};

  data_type pos[2];

};

struct Random {

  Random() : rrand{} {};

  data_type rrand[2];

};

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  
  std::cout << "\nParticle Diffusion example...\n\n";
  const size_t grid_size = 22;
  const size_t n_particles = 256;
  const size_t n_iterations = 10000;
  const size_t n_moves = n_particles * n_iterations;
  const size_t gs2 = grid_size * grid_size;
  const size_t planes = 3;
  const size_t p_size = n_particles, r_size = n_moves, g_size= gs2*planes;
  const float center = grid_size / 2, radius= 0.5;

  Particle * p    = memoryManager::allocate<Particle>( p_size );
  Random * r      = memoryManager::allocate<Random>( r_size );
  size_t * grid   = memoryManager::allocate<size_t>( g_size );

  // _matmult_ranges_start
  //RAJA::RangeSegment particle_range(0, p_size);
  //RAJA::RangeSegment random_range(0, r_size);
  RAJA::RangeSegment particle_range(0, 0);
  RAJA::RangeSegment random_range(0, 0);

  // _matmult_ranges_end

  // _matmult_views_start
  RAJA::View<Particle, RAJA::Layout<1>> pview(p,    p_size);
  RAJA::View<Random,   RAJA::Layout<1>> rview(r,    r_size);
  RAJA::View<size_t,   RAJA::Layout<1>> gview(grid, g_size);
  // _matmult_views_end
  //RAJA::TypedIndexSet< RAJA::RangeSegment, RAJA::RangeSegment > iset;
  //iset.push_back(particle_range);
  //iset.push_back(RAJA::RangeSegment(0,3) );
  auto omp_particle_init = [=] (size_t p_index) 
  {
    pview(p_index).pos[0] = center;
    pview(p_index).pos[1] = center;
  };

  auto omp_random_init = [=] (size_t p_index) 
  {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine 
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1.0);

    rview(p_index).rrand[0] = dis(gen);
    rview(p_index).rrand[1] = dis(gen);
    
  };

  auto cuda_particle_init = [=] RAJA_DEVICE (size_t p_index) 
  {
    pview(p_index).pos[0] = center;
    pview(p_index).pos[1] = center;
  };

  auto cuda_random_init = [=] RAJA_DEVICE (int p_index) 
  {
     curandStatePhilox4_32_10_t state;
     curand_init(1234, 0, p_index, &state);

     rview(p_index).rrand[0] = curand_uniform(&state);
     rview(p_index).rrand[1] = curand_uniform(&state);
     
  };

  auto omp_compute = [=] (size_t p_index )
  {
    Particle& p = pview(p_index);

    bool inside_cell = false;
    // Coordinates of the last known cell this particle resided i
    unsigned int prev_known_cell_coordinate_X;
    unsigned int prev_known_cell_coordinate_Y;

    //Each particle performs this loop
    for( size_t iter = 0; iter < n_iterations; ++iter )
    {
      float displacement_X = rview(iter * n_particles + p_index).rrand[0];
      float displacement_Y = rview(iter * n_particles + p_index).rrand[1];

      p.pos[0] += displacement_X;
      p.pos[1] += displacement_Y;
   
      float dX = abs(p.pos[0] - round(p.pos[0] ) );
      float dY = abs(p.pos[1] - round(p.pos[1] ) );

      int iX = floor(p.pos[0] + 0.5);
      int iY = floor(p.pos[1] + 0.5);

      bool increment_C1 = false;
      bool increment_C2 = false;
      bool increment_C3 = false;
      bool decrement_C2_for_previous_cell = false;
      bool update_coordinates = false;

      // Check if particle's grid indices are still inside computation grid
      if ((iX < grid_size) && (iY < grid_size) && (iX >= 0) && (iY >= 0)) {
        // Compare the radius to particle's distance from center of cell
	if (radius >= sqrt(dX * dX + dY * dY)) {
	  increment_C1 = true;
	  if (!inside_cell) {
	    increment_C2 = true;
	    increment_C3 = true;
	    inside_cell = true;
	    update_coordinates = true;
	  }
	  else if (prev_known_cell_coordinate_X != iX ||
	           prev_known_cell_coordinate_Y != iY) {
            increment_C2 = true;
	    increment_C3 = true;
	    update_coordinates = true;
	    decrement_C2_for_previous_cell = true;
	  }
	}
        else if (inside_cell) {
	  inside_cell = false;
          decrement_C2_for_previous_cell = true;
        }
      }	
      else if (inside_cell) {
        inside_cell = false;
	decrement_C2_for_previous_cell = true;
      }

      size_t layer;
      size_t curr_coordinates = iX + iY * grid_size;
      size_t prev_coordinates = prev_known_cell_coordinate_X +
	                        prev_known_cell_coordinate_Y * grid_size;

      layer = gs2;


      if (decrement_C2_for_previous_cell)
      {
        size_t * g_prev = &gview(prev_coordinates + layer);	
        RAJA::AtomicRef<size_t, RAJA::omp_atomic> ag_prev(g_prev);
        ag_prev--;	
        //atomic_fetch_sub<size_t>(gview(prev_coordinates + layer), 1);

      }
      if (update_coordinates) {
        prev_known_cell_coordinate_X = iX;
	prev_known_cell_coordinate_Y = iY;
      }


      layer = 0;
      if (increment_C1)
      {
        size_t * g_curr = &gview(curr_coordinates + layer);	
        RAJA::AtomicRef<size_t, RAJA::omp_atomic> ag_curr(g_curr);
	ag_curr++;
        //atomic_fetch_add<size_t>(gview(curr_coordinates + layer), 1);
      }
  
      layer = gs2;
      if (increment_C2)
      {
        size_t * g_curr = &gview(curr_coordinates + layer);	
        RAJA::AtomicRef<size_t, RAJA::omp_atomic> ag_curr(g_curr);
	ag_curr++;
        //atomic_fetch_add<size_t>(gview(curr_coordinates + layer), 1);
      }

      layer = gs2 + gs2;
      if (increment_C3)
      {
        size_t * g_curr = &gview(curr_coordinates + layer);	
        RAJA::AtomicRef<size_t, RAJA::omp_atomic> ag_curr(g_curr);
	ag_curr++;
        //atomic_fetch_add<size_t>(gview(curr_coordinates + layer), 1);
      }


    }

  };

  auto cuda_compute = [=] RAJA_DEVICE (size_t p_index )
  {
    Particle& p = pview(p_index);

    bool inside_cell = false;
    // Coordinates of the last known cell this particle resided i
    unsigned int prev_known_cell_coordinate_X;
    unsigned int prev_known_cell_coordinate_Y;

    //Each particle performs this loop
    for( size_t iter = 0; iter < n_iterations; ++iter )
    {
      float displacement_X = rview(iter * n_particles + p_index).rrand[0];
      float displacement_Y = rview(iter * n_particles + p_index).rrand[1];

      p.pos[0] += displacement_X;
      p.pos[1] += displacement_Y;
   
      float dX = abs(p.pos[0] - round(p.pos[0] ) );
      float dY = abs(p.pos[1] - round(p.pos[1] ) );

      int iX = floor(p.pos[0] + 0.5);
      int iY = floor(p.pos[1] + 0.5);

      bool increment_C1 = false;
      bool increment_C2 = false;
      bool increment_C3 = false;
      bool decrement_C2_for_previous_cell = false;
      bool update_coordinates = false;

      // Check if particle's grid indices are still inside computation grid
      if ((iX < grid_size) && (iY < grid_size) && (iX >= 0) && (iY >= 0)) {
        // Compare the radius to particle's distance from center of cell
	if (radius >= sqrt(dX * dX + dY * dY)) {
	  increment_C1 = true;
	  if (!inside_cell) {
	    increment_C2 = true;
	    increment_C3 = true;
	    inside_cell = true;
	    update_coordinates = true;
	  }
	  else if (prev_known_cell_coordinate_X != iX ||
	           prev_known_cell_coordinate_Y != iY) {
            increment_C2 = true;
	    increment_C3 = true;
	    update_coordinates = true;
	    decrement_C2_for_previous_cell = true;
	  }
	}
        else if (inside_cell) {
	  inside_cell = false;
          decrement_C2_for_previous_cell = true;
        }
      }	
      else if (inside_cell) {
        inside_cell = false;
	decrement_C2_for_previous_cell = true;
      }

      size_t layer;
      size_t curr_coordinates = iX + iY * grid_size;
      size_t prev_coordinates = prev_known_cell_coordinate_X +
	                        prev_known_cell_coordinate_Y * grid_size;

      layer = gs2;


      if (decrement_C2_for_previous_cell)
        RAJA::atomicDec<RAJA::cuda_atomic>( &gview(prev_coordinates + layer) );

      if (update_coordinates) {
        prev_known_cell_coordinate_X = iX;
	prev_known_cell_coordinate_Y = iY;
      }


      layer = 0;
      if (increment_C1)
        RAJA::atomicInc<RAJA::cuda_atomic>( &gview(curr_coordinates + layer) );
  
      layer = gs2;
      if (increment_C2)
        RAJA::atomicInc<RAJA::cuda_atomic>( &gview(curr_coordinates + layer) );

      layer = gs2 + gs2;
      if (increment_C3)
        RAJA::atomicInc<RAJA::cuda_atomic>( &gview(curr_coordinates + layer) );

    }

  }; 

  ////////////////////////////////////////////////////////////////////////////////////
  auto start = std::chrono::system_clock::now();
  RAJA::forall<RAJA::omp_parallel_for_exec>(particle_range, omp_particle_init);
  RAJA::forall<RAJA::omp_parallel_for_exec>(random_range,   omp_random_init);
  RAJA::forall<RAJA::omp_parallel_for_exec>(particle_range, omp_compute);
  auto end  = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "OMP elapsed time : "<<  elapsed.count() << "us" << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////

  start = std::chrono::system_clock::now();
  RAJA::forall<RAJA::cuda_exec<256> >(particle_range, cuda_particle_init);
  RAJA::forall<RAJA::cuda_exec<256> >(random_range,   cuda_random_init);
  RAJA::forall<RAJA::cuda_exec<256> >(particle_range, cuda_compute);
  
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "CUDA elapsed time : "<<  elapsed.count() << "us\n\n";
  ////////////////////////////////////////////////////////////////////////////////////
}
