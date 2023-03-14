// mtrand.cpp, see include file mtrand.h for information

#include "mtrand.h"
// non-inline function definitions and static member definitions cannot
// reside in header file because of the risk of multiple declarations

// initialization of static private members
unsigned long MTRand_int32::state[n] = { 0x0UL };
int MTRand_int32::p = 0;
bool MTRand_int32::init = false;

void MTRand_int32::gen_state() { // generate new state vector
	for (int i = 0; i < (n - m); ++i)
		state[i] = state[i + m] ^ twiddle(state[i], state[i + 1]);
	for (int i = n - m; i < (n - 1); ++i)
		state[i] = state[i + m - n] ^ twiddle(state[i], state[i + 1]);
	state[n - 1] = state[m - 1] ^ twiddle(state[n - 1], state[0]);
	p = 0; // reset position
}

void MTRand_int32::seed(unsigned long s) {  // init by 32 bit seed
	state[0] = s & 0xFFFFFFFFUL; // for > 32 bit machines
	for (int i = 1; i < n; ++i) {
		state[i] = 1812433253UL * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
		// see Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier
		// in the previous versions, MSBs of the seed affect only MSBs of the array state
		// 2002/01/09 modified by Makoto Matsumoto
		state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
	}
	p = n; // force gen_state() to be called for next random number
}

void MTRand_int32::seed(const unsigned long* array, int size) { // init by array
	seed(19650218UL);
	int i = 1, j = 0;
	for (int k = ((n > size) ? n : size); k; --k) {
		state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525UL))
			+ array[j] + j; // non linear
		state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
		++j; j %= size;
		if ((++i) == n) { state[0] = state[n - 1]; i = 1; }
	}
	for (int k = n - 1; k; --k) {
		state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941UL)) - i;
		state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
		if ((++i) == n) { state[0] = state[n - 1]; i = 1; }
	}
	state[0] = 0x80000000UL; // MSB is 1; assuring non-zero initial array
	p = n; // force gen_state() to be called for next random number
}


// Gauss random-number generator //
namespace nse
{
	GaussRand::GaussRand() : seed(0), mean((double)0), variance((double)0) {
	}
	GaussRand::~GaussRand() {
	}
	GaussRand::GaussRand(const GaussRand& gen) :
		mean(gen.mean), variance(gen.variance),
		seed(gen.seed)
	{
		mt.seed(seed);
	}

	void GaussRand::set(const double _mean, const double _variance,
		const long int _seed)
	{
		seed = _seed;
		mean = _mean; variance = _variance;

		mt.seed(seed);
	}

	double GaussRand::s_rand()
	{
		double u, v;
		double sqr_sum;

		do
		{
			u = 2 * uni_rand() - 1;
			v = 2 * uni_rand() - 1;

			sqr_sum = u * u + v * v;
		} while (sqr_sum >= (double) 1.0);

		return mean + sqrt(variance) * sqrt(-(double)2.0 * log(sqr_sum) / sqr_sum) * u;
	}

	double GaussRand::mt_rand()
	{
		double u, v;
		double sqr_sum;

		do
		{
			u = 2 * mt() - 1;
			v = 2 * mt() - 1;

			sqr_sum = u * u + v * v;
		} while (sqr_sum >= (double) 1.0);

		return mean + sqrt(variance) * sqrt(-(double)2.0 * log(sqr_sum) / sqr_sum) * u;
	}

	double GaussRand::uni_rand()
	{
		const long int a = 48271;
		const long int m = 2147483647;
		const long int q = (m / a);
		const long int r = (m % a);

		long int hi = seed / q;
		long int lo = seed % q;
		long int test = a * lo - r * hi;
		if (test > 0)
			seed = test;
		else
			seed = test + m;
		return (double)seed / m;
	}
}