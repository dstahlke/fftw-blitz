#include "fftw_blitz.h"

#if FFTWBLITZ_MT
	boost::mutex fftw_alloc_mutex;
#endif
