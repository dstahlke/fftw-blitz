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
		// Supposedly std::complex is always compatible with fftw's 
		// custom complex type (which is double[2])
		// http://www.fftw.org/doc/Complex-numbers.html
		blitz_array(
			reinterpret_cast<std::complex<double> *>(fftw_mem.ptr), 
			shape, 
			blitz::neverDeleteData)
	{ }
};

template <int DIM, class INBUF, class INRET, class OUTBUF, class OUTRET>
class FFTW_Base : public boost::noncopyable {
	friend class FFTW_R2C_2D;
	friend class FFTW_C2R_2D;
	friend class FFTW_R2C_1D;
	friend class FFTW_C2R_1D;

	INBUF in;
	OUTBUF out;
	fftw_plan plan;

	FFTW_Base(
		blitz::TinyVector<int, DIM> in_shape,
		blitz::TinyVector<int, DIM> out_shape
	) :
		in(in_shape), out(out_shape), plan(0)
	{ }

	~FFTW_Base() {
		LOCK_FFTW_ALLOC_MUTEX();
		fftw_destroy_plan(plan);
	}

public:
	inline INRET &input() { return in.blitz_array; }
	inline OUTRET &output() { return out.blitz_array; }
	inline void execute() { fftw_execute(plan); }

	template <class T> // template here allows usage of blitz expressions
	inline OUTRET execute(T in) {
		input() = in;
		execute();
		return output();
	}
};

typedef FFTW_Base<2, 
	FFTW_Blitz_Real<2>, blitz::Array<double, 2>,
	FFTW_Blitz_Cplx<2>, blitz::Array<std::complex<double>, 2>
	> FFTW_R2C_2D_Base;

class FFTW_R2C_2D : public FFTW_R2C_2D_Base {
public:
	FFTW_R2C_2D(int size0, int size1, unsigned int flags=FFTW_ESTIMATE) :
		FFTW_R2C_2D_Base(
			blitz::shape(size0, size1),
			blitz::shape(size0, (size1/2+1))
		)
	{
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_r2c_2d(
			size0, size1, 
			in.fftw_mem.ptr, out.fftw_mem.ptr, 
			flags);
	}
};

typedef FFTW_Base<2, 
	FFTW_Blitz_Cplx<2>, blitz::Array<std::complex<double>, 2>,
	FFTW_Blitz_Real<2>, blitz::Array<double, 2>
	> FFTW_C2R_2D_Base;

class FFTW_C2R_2D : public FFTW_C2R_2D_Base {
public:
	FFTW_C2R_2D(int size0, int size1, unsigned int flags=FFTW_ESTIMATE) :
		FFTW_C2R_2D_Base(
			blitz::shape(size0, (size1/2+1)),
			blitz::shape(size0, size1)
		)
	{
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_c2r_2d(
			size0, size1, 
			in.fftw_mem.ptr, out.fftw_mem.ptr, 
			flags);
	}
};

typedef FFTW_Base<1, 
	FFTW_Blitz_Real<1>, blitz::Array<double, 1>,
	FFTW_Blitz_Cplx<1>, blitz::Array<std::complex<double>, 1>
	> FFTW_R2C_1D_Base;

class FFTW_R2C_1D : public FFTW_R2C_1D_Base {
public:
	FFTW_R2C_1D(int size, unsigned int flags=FFTW_ESTIMATE) :
		FFTW_R2C_1D_Base(
			blitz::shape(size),
			blitz::shape((size/2+1))
		)
	{
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_r2c_1d(
			size, 
			in.fftw_mem.ptr, out.fftw_mem.ptr, 
			flags);
	}
};

typedef FFTW_Base<1, 
	FFTW_Blitz_Cplx<1>, blitz::Array<std::complex<double>, 1>,
	FFTW_Blitz_Real<1>, blitz::Array<double, 1>
	> FFTW_C2R_1D_Base;

class FFTW_C2R_1D : public FFTW_C2R_1D_Base {
public:
	FFTW_C2R_1D(int size, unsigned int flags=FFTW_ESTIMATE) :
		FFTW_C2R_1D_Base(
			blitz::shape((size/2+1)),
			blitz::shape(size)
		)
	{
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_c2r_1d(
			size, 
			in.fftw_mem.ptr, out.fftw_mem.ptr, 
			flags);
	}
};

#endif // FFTW_BLITZ_H
