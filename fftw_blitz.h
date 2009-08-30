#ifndef FFTW_BLITZ_H
#define FFTW_BLITZ_H

#include <fftw3.h>
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>
#include <boost/utility.hpp>

namespace fftwblitz {
	// these are defined for convenience and to ease the possibility
	// of using something other than blitz in the future (such as uBLAS)
	typedef blitz::Array<double, 1> real1d;
	typedef blitz::Array<std::complex<double>, 1> cplx1d;
	typedef blitz::TinyVector<int, 1> shape1d;
	typedef blitz::Array<double, 2> real2d;
	typedef blitz::Array<std::complex<double>, 2> cplx2d;
	typedef blitz::TinyVector<int, 2> shape2d;
}

class FFTW_R2C_2D : public boost::noncopyable {
	double *spatial;
	fftw_complex *freq;
	fftw_plan plan;
	// these are pointers so we can construct them in the
	// body of our constructor
	fftwblitz::real2d *spatial_view;
	fftwblitz::cplx2d *freq_view;

	void init(int size0, int size1, unsigned int flags);

public:
	FFTW_R2C_2D(fftwblitz::shape2d size, unsigned int flags=FFTW_ESTIMATE);
	FFTW_R2C_2D(int size0, int size1, unsigned int flags=FFTW_ESTIMATE);

	~FFTW_R2C_2D();

	fftwblitz::real2d &input();
	fftwblitz::cplx2d &output();
	void execute();

	template <class T> // template here allows usage of blitz expressions
	inline fftwblitz::cplx2d forward(T in) {
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
	fftwblitz::real2d *spatial_view;
	fftwblitz::cplx2d *freq_view;

	void init(int size0, int size1, unsigned int flags);

public:
	FFTW_C2R_2D(fftwblitz::shape2d size, unsigned int flags=FFTW_ESTIMATE);
	FFTW_C2R_2D(int size0, int size1, unsigned int flags=FFTW_ESTIMATE);

	~FFTW_C2R_2D();

	fftwblitz::cplx2d &input();
	fftwblitz::real2d &output();
	void execute();

	template <class T> // template here allows usage of blitz expressions
	inline fftwblitz::real2d inverse(T in) {
		input() = in;
		execute();
		return output();
	}
};

class FFTW_R2C_1D : public boost::noncopyable {
	double *spatial;
	fftw_complex *freq;
	fftw_plan plan;
	// these are pointers so we can construct them in the
	// body of our constructor
	fftwblitz::real1d *spatial_view;
	fftwblitz::cplx1d *freq_view;

	void init(int size, unsigned int flags);

public:
	FFTW_R2C_1D(fftwblitz::shape1d size, unsigned int flags=FFTW_ESTIMATE);
	FFTW_R2C_1D(int size, unsigned int flags=FFTW_ESTIMATE);

	~FFTW_R2C_1D();

	fftwblitz::real1d &input();
	fftwblitz::cplx1d &output();
	void execute();

	template <class T> // template here allows usage of blitz expressions
	inline fftwblitz::cplx1d forward(T in) {
		input() = in;
		execute();
		return output();
	}
};

class FFTW_C2R_1D : public boost::noncopyable {
	double *spatial;
	fftw_complex *freq;
	fftw_plan plan;
	// these are pointers so we can construct them in the
	// body of our constructor
	fftwblitz::real1d *spatial_view;
	fftwblitz::cplx1d *freq_view;

	void init(int size, unsigned int flags);

public:
	FFTW_C2R_1D(fftwblitz::shape1d size, unsigned int flags=FFTW_ESTIMATE);
	FFTW_C2R_1D(int size, unsigned int flags=FFTW_ESTIMATE);

	~FFTW_C2R_1D();

	fftwblitz::cplx1d &input();
	fftwblitz::real1d &output();
	void execute();

	template <class T> // template here allows usage of blitz expressions
	inline fftwblitz::real1d inverse(T in) {
		input() = in;
		execute();
		return output();
	}
};

#endif // FFTW_BLITZ_H
