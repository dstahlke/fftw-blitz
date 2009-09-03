#include "fftw_blitz.h"

#if _REENTRANT
	boost::mutex fftw_alloc_mutex;
#endif
