#ifndef FFTW_BLITZ_H
#define FFTW_BLITZ_H

#include <fftw3.h>
#include <boost/utility.hpp>

#include "common.h"

class FFTW_R2C_2D : public boost::noncopyable {
	double *spatial;
	fftw_complex *freq;
	fftw_plan plan;
	// these are pointers so we can construct them in the
	// body of our constructor
	grid_t *spatial_view;
	grid_cplx_t *freq_view;

public:
	FFTW_R2C_2D(yx_t size);

	~FFTW_R2C_2D();

	grid_t &input();
	grid_cplx_t &output();
	void execute();

	template <class T> // template here allows usage of blitz expressions
	inline grid_cplx_t forward(T in) {
		input() = in;
		execute();
		return output();
	}
};

class FFTW_C2R_2D : public boost::noncopyable {
	double *spatial;
	fftw_complex *freq;
	fftw_plan plan;
	// these are pointers so we can construct them in the
	// body of our constructor
	grid_t *spatial_view;
	grid_cplx_t *freq_view;

public:
	FFTW_C2R_2D(yx_t size);

	~FFTW_C2R_2D();

	grid_cplx_t &input();
	grid_t &output();
	void execute();

	template <class T> // template here allows usage of blitz expressions
	inline grid_t inverse(T in) {
		input() = in;
		execute();
		return output();
	}
};

#endif // FFTW_BLITZ_H
