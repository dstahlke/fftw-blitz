/*
	Copyright 2010 Daniel Stahlke

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

/** \file fftw_blitz.cc
 * \brief C++ wrapper for FFTW, using Blitz++ arrays.
 *
 * This file can be downloaded from http://www.stahlke.org/dan/fftw-blitz
 */

#include "fftw_blitz.h"

#if _REENTRANT
	/** \brief Mutex to ensure that there will be no simultaneous usage
	 * of non-thread-safe FFTW functions.
	 *
	 * http://www.fftw.org/fftw3_doc/Thread-safety.html
	 */
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
