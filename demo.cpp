#include <iostream>
#include "fftw_blitz.h"

void test_1d() {
	int size = 10;
	blitz::Array<double, 1>  in(size);
	FFTW_R2C_1D fwd(size);
	FFTW_C2R_1D inv(size);

	blitz::firstIndex i;
	in = 10.0 
		+ 20.0*sin(i / double(size) * 2.0 * M_PI * 3.0)
		+ 30.0*cos(i / double(size) * 2.0 * M_PI * 4.0);
	blitz::Array<std::complex<double>, 1> out(fwd.output().shape());
	out = fwd.forward(in) / double(size);
	out = where(real(out*conj(out)) > 1e-9, out, 0);
	std::cout << out << std::endl;

	blitz::Array<double, 1> err(size);
	err = in*size - inv.inverse(fwd.forward(in));
	std::cout << "err=" << max(fabs(err)) << std::endl;
}

void test_2d() {
	int size_y=10, size_x=8;
	blitz::Array<double, 2>  in(size_y, size_x);
	FFTW_R2C_2D fwd(size_y, size_x);
	FFTW_C2R_2D inv(size_y, size_x);

	blitz::firstIndex i;
	blitz::secondIndex j;
	in = 10.0 
		+ 20.0*sin(i / double(size_y) * 2.0 * M_PI * 3.0)
		+ 30.0*cos(i / double(size_y) * 2.0 * M_PI * 2.0)
		      *sin(j / double(size_x) * 2.0 * M_PI * 3.0);
	blitz::Array<std::complex<double>, 2> out(fwd.output().shape());
	out = fwd.execute(in) / double(size_y*size_x);
	out = where(real(out*conj(out)) > 1e-9, out, 0);
	std::cout << out << std::endl;

	blitz::Array<double, 2> err(in.shape());
	err = in*size_y*size_x - inv.execute(fwd.execute(in));
	std::cout << "err=" << max(fabs(err)) << std::endl;
}

int main() {
	test_1d();
	test_2d();
}
