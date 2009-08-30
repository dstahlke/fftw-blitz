#include "fftw_blitz.h"

// FIXME - these are needed if threads are to be used
static void acquire_fftw_mutex() { }
static void release_fftw_mutex() { }

FFTW_R2C_2D::FFTW_R2C_2D(yx_t size) {
	int size_y = size[0];
	int size_x = size[1];

	acquire_fftw_mutex();
	spatial = (double *)fftw_malloc(sizeof(double) * size_y * size_x);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size_y * (size_x/2+1));
	plan = fftw_plan_dft_r2c_2d(size_y, size_x, spatial, freq, FFTW_ESTIMATE);
	release_fftw_mutex();

	spatial_view = new grid_t(
		spatial, yx_t(size_y, size_x), blitz::neverDeleteData);
	// Supposedly std::complex is always compatible with fftw's 
	// custom complex type (which is double[2])
	// http://www.fftw.org/doc/Complex-numbers.html
	// FIXME - how to write this cast in C++?
	freq_view = new grid_cplx_t(
		(std::complex<double> *)freq, yx_t(size_y, (size_x/2+1)), blitz::neverDeleteData);
}

FFTW_R2C_2D::~FFTW_R2C_2D() {
	acquire_fftw_mutex();
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
	release_fftw_mutex();
}

grid_t &FFTW_R2C_2D::input() {
	return *spatial_view; 
}

grid_cplx_t &FFTW_R2C_2D::output() {
	return *freq_view; 
}

void FFTW_R2C_2D::execute() {
	fftw_execute(plan); 
}

///// FFTW_C2R_2D /////

FFTW_C2R_2D::FFTW_C2R_2D(yx_t size) {
	int size_y = size[0];
	int size_x = size[1];

	acquire_fftw_mutex();
	spatial = (double *)fftw_malloc(sizeof(double) * size_y * size_x);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size_y * (size_x/2+1));
	plan = fftw_plan_dft_c2r_2d(size_y, size_x, freq, spatial, FFTW_ESTIMATE);
	release_fftw_mutex();

	spatial_view = new grid_t(
		spatial, yx_t(size_y, size_x), blitz::neverDeleteData);
	// Supposedly std::complex is always compatible with fftw's 
	// custom complex type (which is double[2])
	// http://www.fftw.org/doc/Complex-numbers.html
	// FIXME - how to write this cast in C++?
	freq_view = new grid_cplx_t(
		(std::complex<double> *)freq, yx_t(size_y, (size_x/2+1)), blitz::neverDeleteData);
}

FFTW_C2R_2D::~FFTW_C2R_2D() {
	acquire_fftw_mutex();
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
	release_fftw_mutex();
}

grid_cplx_t &FFTW_C2R_2D::input() {
	return *freq_view; 
}

grid_t &FFTW_C2R_2D::output() {
	return *spatial_view; 
}

void FFTW_C2R_2D::execute() {
	fftw_execute(plan); 
}
