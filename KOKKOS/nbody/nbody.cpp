#include <chrono>
#include <numeric>
#include <random>

#include <Kokkos_Core.hpp>
#include <Kokkos_Cuda.hpp>
#include <Kokkos_DualView.hpp>

#include <cstdio>

// See 01_thread_teams for an explanation of a basic TeamPolicy
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

struct Particle
{

  __host__ __device__
  Particle(){}

  float pos[3];
  float vel[3];
  float acc[3];
  float mass;
};

struct compute {};
struct reduc   {};

//template<typename ExecSpace = Kokkos::Cuda, typename MemSpace = Kokkos::CudaUVMSpace>
template<typename ExecSpace, typename MemSpace>
struct GSimulation {

  using ParticleView   = typename Kokkos::DualView<Particle *, Kokkos::Device<ExecSpace, MemSpace> >;
  using EnergyView     = typename Kokkos::DualView<float *, Kokkos::Device<ExecSpace, MemSpace> >;

  GSimulation( )
  : _nparts( 1600 ), nsteps_(10000), time_step_(0.1), particles_("particle_view", 1600), energyV("energy", 1)
  {

    init_points();	  
    
    auto start = high_resolution_clock::now();
    for( int s = 1; s <= nsteps_; s++)
    {
      Kokkos::parallel_for("Stage1", Kokkos::RangePolicy<compute, ExecSpace>(0, _nparts), *this); 
      Kokkos::parallel_for("Reduc1", Kokkos::RangePolicy<reduc,   ExecSpace>(0, _nparts), *this); 
      //Kokkos::parallel_reduce(Kokkos::RangePolicy<reduc, ExecSpace>(0, _nparts), *this, Kokkos::Sum<float, MemSpace>(energy_) );

      auto kenergy = 0.5 * energy_;
    }

    auto end = high_resolution_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    std::cout << "Elapsed Time : " << diff.count() << " ms\n";
        
  }

  void init_points()
  {
    for(size_t i=0; i < _nparts; i++)
    {
      auto p_h = particles_.h_view;
      Particle& p = p_h(i);

      std::random_device rd;  // random number generator
      std::mt19937 gen(42);
      {
        std::uniform_real_distribution<float> unif_d(0, 1.0);
        p.pos[0] = unif_d(gen);
        p.pos[1] = unif_d(gen);
        p.pos[2] = unif_d(gen);
        p.mass   = _nparts * unif_d(gen);
      }
      {
        std::uniform_real_distribution<float> unif_d(-1.0, 1.0);
        p.vel[0] = unif_d(gen) * 1.0e-3f;
        p.vel[1] = unif_d(gen) * 1.0e-3f;
        p.vel[2] = unif_d(gen) * 1.0e-3f;
      }
      {
        p.acc[0] = 0.f;
        p.acc[1] = 0.f;
        p.acc[2] = 0.f;
      } 
    
    }

    particles_.modify_device();
    particles_.sync_device();
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const compute&, const int i ) const {
    Particle& p = particles_.d_view(i);
    float acc0 = p.acc[0];
    float acc1 = p.acc[1];
    float acc2 = p.acc[2];

    for(int j=0; j< _nparts; j++)
    {
      float dx, dy, dz;
      float distance_sqr = 0.0f;
      float distance_inv = 0.0f;

      dx = p.pos[0] - p.pos[0];  // 1flop
      dy = p.pos[1] - p.pos[1];  // 1flop
      dz = p.pos[2] - p.pos[2];  // 1flop

      distance_sqr =  dx * dx + dy * dy + dz * dz + kSofteningSquared;
      distance_inv = 1.0f / sqrt(distance_sqr);

      acc0 += dx * kG * p.mass * distance_inv * distance_inv *distance_inv;
      acc1 += dy * kG * p.mass * distance_inv * distance_inv *distance_inv;
      acc2 += dz * kG * p.mass * distance_inv * distance_inv * distance_inv;
    }

    p.acc[0] = acc0;
    p.acc[1] = acc1;
    p.acc[2] = acc2;

  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const reduc &, const int& i ) const {

    float dt = time_step_;
    Particle& p = particles_.d_view(i);
    p.vel[0] += p.acc[0] * dt;
    p.vel[1] += p.acc[1] * dt;
    p.vel[2] += p.acc[2] * dt;

    p.pos[0] += p.vel[0] * dt;
    p.pos[1] += p.vel[1] * dt;
    p.pos[2] += p.vel[2] * dt;

    p.acc[0] = 0.f;
    p.acc[1] = 0.f;
    p.acc[2] = 0.f;

    float& e = energyV.d_view(0);

    float energy = (p.mass *
    	           (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1] +
	            p.vel[2] * p.vel[2]));

    Kokkos::atomic_fetch_add(&e, energy);
    
  }


  //KOKKOS_INLINE_FUNCTION
  ~GSimulation()
  { 
    
  }


  void complete()
  {
    printf("Compeleted...\n");
  }  

  int nsteps_;
  size_t _nparts;
  float energy_;
  EnergyView energyV;
  float time_step_;
  ParticleView particles_;
  const float kSofteningSquared = 1e-3f;
  const float kG = 6.67259e-11f;
};


int main(int narg, char* args[]) {
  Kokkos::initialize(narg, args);
  std::cout << "Hello World" << std::endl;
  //size_t n_parts = 1600;
  {
    std::cout << "Running CUDA benchmark" << std::endl;
    GSimulation<Kokkos::Cuda, Kokkos::CudaUVMSpace> gsim;
    gsim.complete();
  }

  {
    std::cout << "Running CPU benchmark" << std::endl;
    GSimulation<Kokkos::OpenMP, Kokkos::HostSpace> gsim;
    gsim.complete();
  }


  std::cout << "Goodbye ..." << std::endl;
  Kokkos::finalize();
}
