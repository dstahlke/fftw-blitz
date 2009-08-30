#include "fftw_blitz.h"

// FIXME - these are needed if threads are to be used
static void acquire_fftw_mutex() { }
static void release_fftw_mutex() { }

///// FFTW_R2C_2D /////

FFTW_R2C_2D::FFTW_R2C_2D(fftwblitz::shape2d size, unsigned int flags) {
	init(size[0], size[1], flags);
}

FFTW_R2C_2D::FFTW_R2C_2D(int size_y, int size_x, unsigned int flags) {
	init(size_y, size_x, flags);
}

void FFTW_R2C_2D::init(int size_y, int size_x, unsigned int flags) {
	acquire_fftw_mutex();
	spatial = (double *)fftw_malloc(sizeof(double) * size_y * size_x);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size_y * (size_x/2+1));
	plan = fftw_plan_dft_r2c_2d(size_y, size_x, spatial, freq, flags);
	release_fftw_mutex();

	spatial_view = new fftwblitz::real2d(
		spatial, blitz::shape(size_y, size_x), blitz::neverDeleteData);
	// Supposedly std::complex is always compatible with fftw's 
	// custom complex type (which is double[2])
	// http://www.fftw.org/doc/Complex-numbers.html
	freq_view = new fftwblitz::cplx2d(
		reinterpret_cast<std::complex<double> *>(freq), 
		blitz::shape(size_y, (size_x/2+1)), blitz::neverDeleteData);
}

FFTW_R2C_2D::~FFTW_R2C_2D() {
	acquire_fftw_mutex();
	delete spatial_view;
	delete freq_view;
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
	release_fftw_mutex();
}

fftwblitz::real2d &FFTW_R2C_2D::input() {
	return *spatial_view; 
}

fftwblitz::cplx2d &FFTW_R2C_2D::output() {
	return *freq_view; 
}

void FFTW_R2C_2D::execute() {
	fftw_execute(plan); 
}

///// FFTW_C2R_2D /////

FFTW_C2R_2D::FFTW_C2R_2D(fftwblitz::shape2d size) {
	init(size[0], size[1]);
}

void FFTW_C2R_2D::init(int size_y, int size_x) {
	acquire_fftw_mutex();
	spatial = (double *)fftw_malloc(sizeof(double) * size_y * size_x);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size_y * (size_x/2+1));
	plan = fftw_plan_dft_c2r_2d(size_y, size_x, freq, spatial, FFTW_ESTIMATE);
	release_fftw_mutex();

	spatial_view = new fftwblitz::real2d(
		spatial, blitz::shape(size_y, size_x), blitz::neverDeleteData);
	// Supposedly std::complex is always compatible with fftw's 
	// custom complex type (which is double[2])
	// http://www.fftw.org/doc/Complex-numbers.html
	freq_view = new fftwblitz::cplx2d(
		reinterpret_cast<std::complex<double> *>(freq), 
		blitz::shape(size_y, (size_x/2+1)), blitz::neverDeleteData);
}

FFTW_C2R_2D::~FFTW_C2R_2D() {
	acquire_fftw_mutex();
	delete spatial_view;
	delete freq_view;
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
	release_fftw_mutex();
}

fftwblitz::cplx2d &FFTW_C2R_2D::input() {
	return *freq_view; 
}

fftwblitz::real2d &FFTW_C2R_2D::output() {
	return *spatial_view; 
}

void FFTW_C2R_2D::execute() {
	fftw_execute(plan); 
}

///// FFTW_R2C_1D /////

FFTW_R2C_1D::FFTW_R2C_1D(fftwblitz::shape1d size) {
	init(size[0]);
}

FFTW_R2C_1D::FFTW_R2C_1D(int size) {
	init(size);
}

void FFTW_R2C_1D::init(int size_x) {
	acquire_fftw_mutex();
	spatial = (double *)fftw_malloc(sizeof(double) * size_x);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_x/2+1));
	plan = fftw_plan_dft_r2c_1d(size_x, spatial, freq, FFTW_ESTIMATE);
	release_fftw_mutex();

	spatial_view = new fftwblitz::real1d(
		spatial, blitz::shape(size_x), blitz::neverDeleteData);
	// Supposedly std::complex is always compatible with fftw's 
	// custom complex type (which is double[2])
	// http://www.fftw.org/doc/Complex-numbers.html
	freq_view = new fftwblitz::cplx1d(
		reinterpret_cast<std::complex<double> *>(freq), 
		blitz::shape(size_x/2+1), blitz::neverDeleteData);
}

FFTW_R2C_1D::~FFTW_R2C_1D() {
	acquire_fftw_mutex();
	delete spatial_view;
	delete freq_view;
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
	release_fftw_mutex();
}

fftwblitz::real1d &FFTW_R2C_1D::input() {
	return *spatial_view; 
}

fftwblitz::cplx1d &FFTW_R2C_1D::output() {
	return *freq_view; 
}

void FFTW_R2C_1D::execute() {
	fftw_execute(plan); 
}

///// FFTW_C2R_1D /////

FFTW_C2R_1D::FFTW_C2R_1D(fftwblitz::shape1d size) {
	init(size[0]);
}

FFTW_C2R_1D::FFTW_C2R_1D(int size) {
	init(size);
}

void FFTW_C2R_1D::init(int size_x) {
	acquire_fftw_mutex();
	spatial = (double *)fftw_malloc(sizeof(double) * size_x);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_x/2+1));
	plan = fftw_plan_dft_c2r_1d(size_x, freq, spatial, FFTW_ESTIMATE);
	release_fftw_mutex();

	spatial_view = new fftwblitz::real1d(
		spatial, blitz::shape(size_x), blitz::neverDeleteData);
	// Supposedly std::complex is always compatible with fftw's 
	// custom complex type (which is double[2])
	// http://www.fftw.org/doc/Complex-numbers.html
	freq_view = new fftwblitz::cplx1d(
		reinterpret_cast<std::complex<double> *>(freq), 
		blitz::shape(size_x/2+1), blitz::neverDeleteData);
}

FFTW_C2R_1D::~FFTW_C2R_1D() {
	acquire_fftw_mutex();
	delete spatial_view;
	delete freq_view;
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
	release_fftw_mutex();
}

fftwblitz::cplx1d &FFTW_C2R_1D::input() {
	return *freq_view; 
}

fftwblitz::real1d &FFTW_C2R_1D::output() {
	return *spatial_view; 
}

void FFTW_C2R_1D::execute() {
	fftw_execute(plan); 
}
