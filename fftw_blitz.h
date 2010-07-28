/*
	Copyright 2009 Daniel Stahlke

	This file is part of fftw-blitz.
	
	fftw-blitz is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	fftw-blitz is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with fftw-blitz.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FFTW_BLITZ_H
#define FFTW_BLITZ_H

/** \file fftw_blitz.h
 * \brief C++ wrapper for FFTW, using Blitz++ arrays.
 */

#include <fftw3.h>
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

#if _REENTRANT
	#include <boost/thread/mutex.hpp>
	// FFTW's memory allocation is not thread-safe.
	// http://www.fftw.org/fftw3_doc/Thread-safety.html
	// This macro will lock a mutex for the remainder of a block scope.
	// For more info, google "boost scoped_lock"
	extern boost::mutex fftw_alloc_mutex;
	#define LOCK_FFTW_ALLOC_MUTEX() boost::mutex::scoped_lock \
		fftw_alloc_lock(fftw_alloc_mutex)
#else
	#define LOCK_FFTW_ALLOC_MUTEX() do{}while(0)
#endif

/** \brief Cast an array of std::complex to FFTW's complex type.
 *
 * Supposedly std::complex is always compatible with fftw's 
 * custom complex type (which is double[2]).
 * http://www.fftw.org/doc/Complex-numbers.html
 */
static inline fftw_complex *fftw_cast_complex(std::complex<double> *arr) {
	return reinterpret_cast<fftw_complex *>(arr);
}

/** \brief A block of memory allocated by FFTW.
 *
 * The constructor will allocate memory in a thread-safe manner and
 * the destructor will free memory.
 */
// This cannot simply be inlined into FFTW_Blitz_Adaptor because we need to
// lock the mutex before construction of the array
template <class T>
class FFTW_Memblock : public boost::noncopyable {
public:
	explicit FFTW_Memblock(size_t size) {
		LOCK_FFTW_ALLOC_MUTEX();
		ptr = reinterpret_cast<T *>(
			fftw_malloc(sizeof(T) * size));
	}

	virtual ~FFTW_Memblock() {
		LOCK_FFTW_ALLOC_MUTEX();
		fftw_free(ptr);
	}

	T *ptr;
};

/** \brief Contains a block of FFTW memory along with a Blitz++ array that
 * references the same memory.
 *
 * \tparam T Type of array elements
 * \tparam N Dimension of array
 */
template <class T, int N>
class FFTW_Blitz_Adaptor : public boost::noncopyable {
public:
	/** \param shape Size of array
	 */
	FFTW_Blitz_Adaptor(blitz::TinyVector<int, N> shape) :
		fftw_mem(product(shape)),
		blitz_array(
			fftw_mem.ptr, shape, 
			blitz::neverDeleteData)
	{ }

	FFTW_Memblock<T> fftw_mem;
	blitz::Array<T, N> blitz_array;
};

/** \brief Base class for FFTW-Blitz adapter classes.
 *
 * \tparam DIM Dimension of array
 * \tparam T_IN Type of array elements for input arrays
 * \tparam T_IN Type of array elements for output arrays
 */
template <int DIM, class T_IN, class T_OUT>
class FFTW_Base : public boost::noncopyable {
protected:
	typedef boost::shared_ptr<FFTW_Blitz_Adaptor<T_IN,  DIM> > in_mem_type;
	typedef boost::shared_ptr<FFTW_Blitz_Adaptor<T_OUT, DIM> > out_mem_type;

	/** Constructor that allocates new memory of the specified size */
	FFTW_Base(
		blitz::TinyVector<int, DIM> in_shape,
		blitz::TinyVector<int, DIM> out_shape
	) :
		in (new FFTW_Blitz_Adaptor<T_IN,  DIM>( in_shape)), 
		out(new FFTW_Blitz_Adaptor<T_OUT, DIM>(out_shape)), 
		plan(NULL)
	{ }

	/** Constructor that uses previously allocated memory */
	FFTW_Base(
		in_mem_type _in,
		out_mem_type _out
	) :
		in (_in), 
		out(_out), 
		plan(NULL)
	{ }

	virtual ~FFTW_Base() {
		LOCK_FFTW_ALLOC_MUTEX();
		if(plan) fftw_destroy_plan(plan);
	}

public:
	/** \brief Returns a reference to the input array */
	inline blitz::Array<T_IN,  DIM> & input() { return in->blitz_array; }
	/** \brief Returns a reference to the output array */
	inline blitz::Array<T_OUT, DIM> &output() { return out->blitz_array; }
	/** \brief Executes the FFT using the input() and output() arrays
	 *
	 * NOTE: It is not allowed to simultaneously execute() the same
	 * FFTW-Blitz adapter object from multiple threads.
	 */
	inline void execute() { fftw_execute(plan); }

	/** \brief Executes the FFT using the given input and returns the output.
	 *
	 * The template parameter allows passing Blitz expressions as input.
	 *
	 * NOTE: this makes use of memory buffers that are stored in this object.
	 * It is therefore not allowed to simultaneously execute() the same
	 * FFTW-Blitz adapter object from multiple threads.
	 */
	template <class T>
	inline blitz::Array<T_OUT, DIM> execute(T data) {
		input() = data;
		execute();
		return output();
	}

protected:
	/** \brief Input memory buffer */
	in_mem_type in;
	/** \brief Output memory buffer */
	out_mem_type out;
	/** \brief The "plan" that is used to execute the FFT */
	fftw_plan plan;
};

typedef FFTW_Base<2, double, std::complex<double> > FFTW_R2C_2D_Base;

/** \brief Adapter for 2-dimensional real-to-complex FFT */
class FFTW_R2C_2D : public FFTW_R2C_2D_Base {
	friend class FFTW_C2R_2D;
public:
	/** \param _size0 Size of input array in x dimension
	  * \param _size1 Size of input array in y dimension
	  * \param _flags Flags to pass to FFTW planner
	  */
	FFTW_R2C_2D(int _size0, int _size1, unsigned int _flags=FFTW_ESTIMATE) :
		FFTW_R2C_2D_Base(
			blitz::shape(_size0, _size1),
			blitz::shape(_size0, (_size1/2+1))
		)
	{
		init(
			_size0,
			_size1,
			_flags
		);
	}

	/** \param _size Size of input array
	  * \param _flags Flags to pass to FFTW planner
	  */
	FFTW_R2C_2D(blitz::TinyVector<int, 2> _size, unsigned int _flags=FFTW_ESTIMATE) :
		FFTW_R2C_2D_Base(
			blitz::shape(_size[0], _size[1]),
			blitz::shape(_size[0], (_size[1]/2+1))
		)
	{
		init(
			_size[0],
			_size[1],
			_flags
		);
	}

	/** \param _in Previously allocated input array
	  * \param _out Previously allocated output array
	  * \param _flags Flags to pass to FFTW planner
	  */
	FFTW_R2C_2D(
		in_mem_type _in,
		out_mem_type _out,
		unsigned int _flags=FFTW_ESTIMATE
	) :
		FFTW_R2C_2D_Base(_in, _out)
	{
		init(
			_in->blitz_array.shape()[0],
			_in->blitz_array.shape()[1],
			_flags
		);
	}

	virtual ~FFTW_R2C_2D();

	/** \brief Execute the inverse FFT, taking data from the output() array and
	 * storing it into the input() array.
	 */
	void executeInverse();

private:
	void init(int size0, int size1, unsigned int flags) {
		inverse = NULL;
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_r2c_2d(
			size0, size1, 
			in->fftw_mem.ptr, 
			fftw_cast_complex(out->fftw_mem.ptr), 
			flags);
	}

	/** \brief Lazily-constructed inverse transform */
	class FFTW_C2R_2D *inverse;
};

typedef FFTW_Base<2, std::complex<double>, double > FFTW_C2R_2D_Base;

/** \brief Adapter for 2-dimensional complex-to-real inverse FFT */
class FFTW_C2R_2D : public FFTW_C2R_2D_Base {
	friend class FFTW_R2C_2D;
public:
	/** \param _size0 Size of input array in x dimension
	  * \param _size1 Size of input array in y dimension
	  * \param _flags Flags to pass to FFTW planner
	  */
	FFTW_C2R_2D(int _size0, int _size1, unsigned int _flags=FFTW_ESTIMATE) :
		FFTW_C2R_2D_Base(
			blitz::shape(_size0, (_size1/2+1)),
			blitz::shape(_size0, _size1)
		)
	{
		init(
			_size0,
			_size1,
			_flags
		);
	}

	/** \param _size Size of input array
	  * \param _flags Flags to pass to FFTW planner
	  */
	FFTW_C2R_2D(blitz::TinyVector<int, 2> _size, unsigned int _flags=FFTW_ESTIMATE) :
		FFTW_C2R_2D_Base(
			blitz::shape(_size[0], (_size[1]/2+1)),
			blitz::shape(_size[0], _size[1])
		)
	{
		init(
			_size[0],
			_size[1],
			_flags
		);
	}

private:
	/** \brief Create instance that shares memory with another instance.
	 *
	 * Used by FFTW_R2C_2D.executeInverse().
	 */
	FFTW_C2R_2D(
		const FFTW_R2C_2D &f,
		unsigned int _flags=FFTW_ESTIMATE
	) :
		FFTW_C2R_2D_Base(f.out, f.in)
	{
		init(
			f.in->blitz_array.shape()[0],
			f.in->blitz_array.shape()[1],
			_flags
		);
	}

	void init(int size0, int size1, unsigned int flags) {
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_c2r_2d(
			size0, size1, 
			fftw_cast_complex(in->fftw_mem.ptr), 
			out->fftw_mem.ptr, 
			flags);
	}
};

typedef FFTW_Base<1, double, std::complex<double> > FFTW_R2C_1D_Base;

/** \brief Adapter for 1-dimensional real-to-complex FFT */
class FFTW_R2C_1D : public FFTW_R2C_1D_Base {
public:
	/** \param size Size of input array
	  * \param flags Flags to pass to FFTW planner
	  */
	FFTW_R2C_1D(int size, unsigned int flags=FFTW_ESTIMATE) :
		FFTW_R2C_1D_Base(
			blitz::shape(size),
			blitz::shape((size/2+1))
		)
	{
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_r2c_1d(
			size, 
			in->fftw_mem.ptr, 
			fftw_cast_complex(out->fftw_mem.ptr), 
			flags);
	}
};

typedef FFTW_Base<1, std::complex<double>, double > FFTW_C2R_1D_Base;

/** \brief Adapter for 1-dimensional complex-to-real inverse FFT */
class FFTW_C2R_1D : public FFTW_C2R_1D_Base {
public:
	/** \param size Size of input array
	  * \param flags Flags to pass to FFTW planner
	  */
	FFTW_C2R_1D(int size, unsigned int flags=FFTW_ESTIMATE) :
		FFTW_C2R_1D_Base(
			blitz::shape((size/2+1)),
			blitz::shape(size)
		)
	{
		LOCK_FFTW_ALLOC_MUTEX();
		plan = fftw_plan_dft_c2r_1d(
			size, 
			fftw_cast_complex(in->fftw_mem.ptr), 
			out->fftw_mem.ptr, 
			flags);
	}
};

#endif // FFTW_BLITZ_H
