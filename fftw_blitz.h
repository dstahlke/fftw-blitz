#ifndef FFTW_BLITZ_H
#define FFTW_BLITZ_H

#include <fftw3.h>
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>
#include <boost/utility.hpp>

#if FFTWBLITZ_MT
	#include <boost/thread/mutex.hpp>
	// FFTW's memory allocation is not thread-safe
	// http://www.fftw.org/fftw3_doc/Thread-safety.html
	extern boost::mutex fftw_alloc_mutex;
	#define LOCK_FFTW_ALLOC_MUTEX() boost::mutex::scoped_lock \
		fftw_alloc_lock(fftw_alloc_mutex)
#else
	#define LOCK_FFTW_ALLOC_MUTEX() do{}while(0)
#endif

// Supposedly std::complex is always compatible with fftw's 
// custom complex type (which is double[2])
// http://www.fftw.org/doc/Complex-numbers.html
#define CAST_CPLX(arr) reinterpret_cast<std::complex<double> *>(arr)

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

template <class T>
class FFTW_Memblock : public boost::noncopyable {
public:
	FFTW_Memblock(size_t size) {
		LOCK_FFTW_ALLOC_MUTEX();
		ptr = reinterpret_cast<T *>(
			fftw_malloc(sizeof(T) * size));
	}

	~FFTW_Memblock() {
		LOCK_FFTW_ALLOC_MUTEX();
		fftw_free(ptr);
	}

	T *ptr;
};

template <int N>
class FFTW_Blitz_Real : public boost::noncopyable {
public:
	FFTW_Memblock<double> fftw_mem;
	blitz::Array<double, N> blitz_array;

	FFTW_Blitz_Real(blitz::TinyVector<int, N> shape) :
		fftw_mem(product(shape)),
		blitz_array(
			fftw_mem.ptr, shape, 
			blitz::neverDeleteData)
	{ }
};

template <int N>
class FFTW_Blitz_Cplx : public boost::noncopyable {
public:
	FFTW_Memblock<fftw_complex> fftw_mem;
	blitz::Array<std::complex<double>, N> blitz_array;

	FFTW_Blitz_Cplx(blitz::TinyVector<int, N> shape) :
		fftw_mem(product(shape)),
		blitz_array(
			reinterpret_cast<std::complex<double> *>(fftw_mem.ptr), 
			shape, 
			blitz::neverDeleteData)
	{ }
};

class FFTW_R2C_2D : public boost::noncopyable {
	FFTW_Blitz_Real<2> in;
	FFTW_Blitz_Cplx<2> out;
	fftw_plan plan;

public:
	FFTW_R2C_2D(int size0, int size1, unsigned int flags=FFTW_ESTIMATE) :
		in(blitz::shape(size0, size1)),
		out(blitz::shape(size0, (size1/2+1)))
	{
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_r2c_2d(
			size0, size1, 
			in.fftw_mem.ptr, out.fftw_mem.ptr, 
			flags);
	}

	~FFTW_R2C_2D() {
		LOCK_FFTW_ALLOC_MUTEX();
		fftw_destroy_plan(plan);
	}

	inline fftwblitz::real2d &input() { return in.blitz_array; }
	inline fftwblitz::cplx2d &output() { return out.blitz_array; }

	void execute() {
		fftw_execute(plan); 
	}

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
