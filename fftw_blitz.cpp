#include "fftw_blitz.h"

#if FFTWBLITZ_MT
	boost::mutex fftw_alloc_mutex;
#endif

///// FFTW_C2R_2D /////

FFTW_C2R_2D::FFTW_C2R_2D(fftwblitz::shape2d size, unsigned int flags) {
	init(size[0], size[1], flags);
}

FFTW_C2R_2D::FFTW_C2R_2D(int size0, int size1, unsigned int flags) {
	init(size0, size1, flags);
}

void FFTW_C2R_2D::init(int size0, int size1, unsigned int flags) {
	LOCK_FFTW_ALLOC_MUTEX();

	spatial = (double *)fftw_malloc(sizeof(double) * size0 * size1);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size0 * (size1/2+1));
	plan = fftw_plan_dft_c2r_2d(size0, size1, freq, spatial, flags);

	spatial_view = new fftwblitz::real2d(
		spatial, 
		blitz::shape(size0, size1), 
		blitz::neverDeleteData);
	freq_view = new fftwblitz::cplx2d(
		CAST_CPLX(freq), 
		blitz::shape(size0, (size1/2+1)), 
		blitz::neverDeleteData);
}

FFTW_C2R_2D::~FFTW_C2R_2D() {
	LOCK_FFTW_ALLOC_MUTEX();

	delete spatial_view;
	delete freq_view;
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
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

FFTW_R2C_1D::FFTW_R2C_1D(fftwblitz::shape1d size, unsigned int flags) {
	init(size[0], flags);
}

FFTW_R2C_1D::FFTW_R2C_1D(int size, unsigned int flags) {
	init(size, flags);
}

void FFTW_R2C_1D::init(int size, unsigned int flags) {
	LOCK_FFTW_ALLOC_MUTEX();

	spatial = (double *)fftw_malloc(sizeof(double) * size);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size/2+1));
	plan = fftw_plan_dft_r2c_1d(size, spatial, freq, flags);

	spatial_view = new fftwblitz::real1d(
		spatial, 
		blitz::shape(size), 
		blitz::neverDeleteData);
	freq_view = new fftwblitz::cplx1d(
		CAST_CPLX(freq), 
		blitz::shape(size/2+1), 
		blitz::neverDeleteData);
}

FFTW_R2C_1D::~FFTW_R2C_1D() {
	LOCK_FFTW_ALLOC_MUTEX();

	delete spatial_view;
	delete freq_view;
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
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

FFTW_C2R_1D::FFTW_C2R_1D(fftwblitz::shape1d size, unsigned int flags) {
	init(size[0], flags);
}

FFTW_C2R_1D::FFTW_C2R_1D(int size, unsigned int flags) {
	init(size, flags);
}

void FFTW_C2R_1D::init(int size, unsigned int flags) {
	LOCK_FFTW_ALLOC_MUTEX();

	spatial = (double *)fftw_malloc(sizeof(double) * size);
	freq = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size/2+1));
	plan = fftw_plan_dft_c2r_1d(size, freq, spatial, flags);

	spatial_view = new fftwblitz::real1d(
		spatial, 
		blitz::shape(size), 
		blitz::neverDeleteData);
	freq_view = new fftwblitz::cplx1d(
		CAST_CPLX(freq), 
		blitz::shape(size/2+1), 
		blitz::neverDeleteData);
}

FFTW_C2R_1D::~FFTW_C2R_1D() {
	LOCK_FFTW_ALLOC_MUTEX();

	delete spatial_view;
	delete freq_view;
	fftw_destroy_plan(plan);
	fftw_free(freq);
	fftw_free(spatial);
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
