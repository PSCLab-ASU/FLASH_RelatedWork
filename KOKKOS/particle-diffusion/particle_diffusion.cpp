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
using std::chrono::microseconds;

struct init{};
struct compute{}; 
struct swap{};


struct Particle
{
   __host__ __device__	
  Particle(){}

  float posX;
  float posY;

};

struct RandMat
{

  __host__ __device__	
  RandMat(){}

  float randX;
  float randY;

};

//template<typename ExecSpace = Kokkos::Cuda, typename MemSpace = Kokkos::CudaUVMSpace>
template<typename ExecSpace, typename MemSpace>
struct GSimulation {

  using ParticleView   = typename Kokkos::View<Particle *,    Kokkos::Device<ExecSpace, MemSpace> >;
  using RandDView      = typename Kokkos::DualView<RandMat *, Kokkos::Device<ExecSpace, MemSpace> >;
  //using RandDView      = typename Kokkos::DualView<RandMat * >;
  using GridView       = typename Kokkos::View<size_t *,      Kokkos::Device<ExecSpace, MemSpace> >;

  GSimulation(size_t planes, size_t n_particles, size_t n_iterations, size_t grid_size, float radius )
  : _planes(planes), _n_particles(n_particles), _n_iterations(n_iterations), _grid_size(grid_size), _radius(radius),
    particles_("particles", n_particles), grid_("grid", grid_size*grid_size*planes), rand_("rand", n_particles*n_iterations)
  {
    //intiialize random matrix
    _init_rand();

    Kokkos::parallel_for("Compute", Kokkos::RangePolicy<init, ExecSpace>(0, _n_particles ), *this); 

    auto start = high_resolution_clock::now();
    Kokkos::parallel_for("Compute", Kokkos::RangePolicy<compute, ExecSpace>(0, _n_particles ), *this); 
    auto end = high_resolution_clock::now();
    auto diff = duration_cast<microseconds>(end - start);

    std::cout << "Elapsed Time : " << diff.count() << " us\n";

      
  }

  void _init_rand()
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d{0,.03};
    
    for (size_t i = 0; i < (size_t)rand_.extent(0); i++) {
      rand_.h_view(i).randX = d(gen);
      rand_.h_view(i).randY = d(gen);
    }

    rand_.modify_device();
    rand_.sync_device();
    //rand_.modify<kokkos::HostSpace::execution_space>();
    //Rand_.sync<ExecSpace>();
  }
 
  KOKKOS_INLINE_FUNCTION
  void operator()(const init &, const int& i ) const {
    particles_(i).posX = _grid_size / 2;   
    particles_(i).posY = _grid_size / 2;   
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const compute &, const int& i ) const {
    Particle& p = particles_(i);
    auto d_rand = rand_.d_view;

    bool inside_cell = false;
    // Coordinates of the last known cell this particle resided in
    unsigned int prev_known_cell_coordinate_X;
    unsigned int prev_known_cell_coordinate_Y;
    //  Motion simulation algorithm
    // --Start iterations--// Each iteration:
    //    1. Updates the position of all particles
    //    2. Checks if particle is inside a cell or not
    //    3. Updates counters in cells array (grid)
    // Each particle performs this loop
    for (size_t iter = 0; iter < _n_iterations; ++iter) {
      // Set the displacements to the random numbers
      float displacement_X = d_rand(iter * _n_particles + i).randX;
      float displacement_Y = d_rand(iter * _n_particles + i).randY;
      // Displace particles
     
      p.posX += displacement_X;
      p.posY += displacement_Y;
      // Compute distances from particle position to grid point i.e.,
      // the particle's distance from center of cell. Subtract the
      // integer value from floating point value to get just the
      // decimal portion. Use this value to later determine if the
      // particle is inside or outside of the cell
      float dX = abs(p.posX - round(p.posX));
      float dY = abs(p.posY - round(p.posY));
      
      int iX = floor(p.posX + 0.5);
      int iY = floor(p.posY + 0.5);

      bool increment_C1 = false;
      bool increment_C2 = false;
      bool increment_C3 = false;
      bool decrement_C2_for_previous_cell = false;
      bool update_coordinates = false;

      // Check if particle's grid indices are still inside computation grid
      if ((iX < _grid_size) && (iY < _grid_size) && (iX >= 0) && (iY >= 0)) {
        // Compare the radius to particle's distance from center of cell
	if (_radius >= sqrt(dX * dX + dY * dY)) 
	{
          // Satisfies counter 1 requirement for cases 1, 3, 4
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
	  // Else: Case 4 --No action required. Counter 1 already updated
	}
        // Case 2b --Particle moved outside grid and outside cell
        else if (inside_cell) 
	{
          inside_cell = false;
	  decrement_C2_for_previous_cell = true;
        }
      }

      // Index variable for 3rd dimension of grid
      size_t layer;
      // Current and previous cell coordinates
      size_t curr_coordinates = iX + iY * _grid_size;
      size_t prev_coordinates = prev_known_cell_coordinate_X +
	                        prev_known_cell_coordinate_Y * _grid_size;

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = _grid_size * _grid_size;

      if (decrement_C2_for_previous_cell)
        Kokkos::atomic_fetch_sub<size_t>(&grid_(prev_coordinates + layer), 1);
      
      if (update_coordinates) {
        prev_known_cell_coordinate_X = iX;
	prev_known_cell_coordinate_Y = iY;
      }

      // Counter 1 layer of the grid (0 * grid_size * grid_size
      layer = 0;
      if (increment_C1)
        Kokkos::atomic_fetch_add<size_t>(&grid_(curr_coordinates + layer), 1);

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = _grid_size * _grid_size;
      if (increment_C2)
	Kokkos::atomic_fetch_add<size_t>(&grid_(curr_coordinates + layer), 1);

      // Counter 3 layer of the grid (2 * grid_size * grid_size)
      layer += layer;
      if (increment_C3)
        Kokkos::atomic_fetch_add<size_t>(&grid_(curr_coordinates + layer), 1);
      
    }

  }

  //KOKKOS_INLINE_FUNCTION
  ~GSimulation()
  { 
    
  }

  void complete()
  {
    printf("Compeleted...\n");
  }  

  float _seed;
  size_t _planes        = 0;
  size_t _n_particles   = 0;
  size_t _n_iterations  = 0;
  size_t _grid_size     = 0;
  float  _radius        = 0;

  ParticleView particles_;
  RandDView rand_;
  GridView  grid_;
};


int main(int narg, char* args[]) {
  Kokkos::initialize(narg, args);
  std::cout << "Hello World" << std::endl;
  float radius=0.5;
  size_t planes=3, n_particles=256, 
	 n_iterations=10000, grid_size=22;

  {
    std::cout << "Running CPU benchmark" << std::endl;
    GSimulation<Kokkos::OpenMP, Kokkos::HostSpace> 
	      gsim(planes, n_particles, n_iterations, grid_size, radius);
    gsim.complete();
  }
  {	  
    std::cout << "Running CUDA benchmark" << std::endl;
    GSimulation<Kokkos::Cuda, Kokkos::CudaUVMSpace>
	      gsim(planes, n_particles, n_iterations, grid_size, radius);
    gsim.complete();
  }


  std::cout << "Goodbye ..." << std::endl;
  Kokkos::finalize();
}
