#include "fftw_blitz.h"

#if _REENTRANT
	boost::mutex fftw_alloc_mutex;
#endif

FFTW_R2C_2D::~FFTW_R2C_2D() {
	delete inverse;
}

void FFTW_R2C_2D::executeInverse() {
	if(!inverse) {
		inverse = new FFTW_C2R_2D(*this);
	}
	inverse->execute();
}
