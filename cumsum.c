#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>

#include <x86intrin.h>


void cumsum_scaler(float *a, int n)
{
	/*
		This function computes an obvious, niave, in-place cumulative sum. It
		does one addition operation per element and every addition opereration
		is dependant on the previous.
	*/
	float s = 0.0;
	while(n > 0)
	{
		s += *a;
		*a++ = s;
		n--;
	}
}

void cumsum_super_scaler(float *a, int n)
{
	/*
		This function computes an in-place cumulative sum.
		It is super-scalar but not SIMD.
		It does 13 addition operations per 8 elements, but has only 4 addition
		operations in the longest chain of dependancies per 8 elements.
	*/
	float s = 0.0, e0, e1, e2, e3, e4, e5, e6, e7, e8;
	while(n >= 8)
	{
		e0 = a[0] + s;
		e1 = a[1];
		e2 = a[2];
		e3 = a[3];
		e4 = a[4];
		e5 = a[5];
		e6 = a[6];
		e7 = a[7];
		
		e1 += e0;
		e3 += e2;
		e5 += e4;
		e7 += e6;

		e2 += e1;
		e3 += e1;
		e6 += e5;
		e7 += e5;

		e4 += e3;
		e5 += e3;
		e6 += e3;
		e7 += e3;

		a[0] = e0;
		a[1] = e1;
		a[2] = e2;
		a[3] = e3;
		a[4] = e4;
		a[5] = e5;
		a[6] = e6;
		a[7] = e7;

		s = e7;
		a += 8;
		n -= 8;
	}
	while(n > 0) 
	{
		e0 = *a;
		*a++ = s = e0 = e0 + s;
		n--;
	}
}

/*
	The palignr instruction works just fine on vectors of floats, but the
	instrinsics available in my version of gcc don't seem to believe this.
	This macro just hides some ugly casting. It depends crusially on the fact
	that casting a __m128 to a __m128i and back works more like this:
		float x = 0.5;
		int y = *(int *)(&x);
		float z = *(float *)(&y);
	Than this:
		float x = 10;
		int y = (int)x;
		float z = (float)y;
	That is, it does not change the bits in the regesters at all, only the
	compiler's attitude towards those bits.
*/
#define _mm_alignr_ps(xmm1, xmm2, imm) ( (__m128) _mm_alignr_epi8( (__m128i)(xmm1), (__m128i)(xmm2), (imm)))

void cumsum_super_scaler_simd(float *a, int n)
{
	/*
		This function computes an in-place cumulative sum.
		It is super scalar and SIMD. 
		It does 88 addition operations per 24 elements, but has only 6 addition
		operations in the longest chain of dependancies per 24 elements.
	*/
	__m128 a0, a1, a2, a3, a4, a5,
			s0, s1, s2, s3, s4, s5,
			b0,
			t, z;
	z = _mm_setzero_ps();
	t = z;

	while(((uint64_t)a & 15) && n > 0)
	{
		a0 = _mm_load_ss(a);
		t = _mm_add_ps(t, a0);
		_mm_store_ss(a, t);
		a++;
		n--;
	}
	while(n >= 24)
	{
		//load values
		a0 = _mm_load_ps(a);
		a1 = _mm_load_ps(a + 4);
		a2 = _mm_load_ps(a + 8);
		a3 = _mm_load_ps(a + 12);
		a4 = _mm_load_ps(a + 16);
		a5 = _mm_load_ps(a + 20);

		//start by adding in running total
		a0 = _mm_add_ps(a0, t);

		//four separate streams of cumsum each done in parallel with the super scaler method
		a1 = _mm_add_ps(a1, a0);
		a3 = _mm_add_ps(a3, a2);
		a5 = _mm_add_ps(a5, a4);
		b0 = a3;
		a4 = _mm_add_ps(a4, b0);
		a5 = _mm_add_ps(a5, b0);
		a2 = _mm_add_ps(a2, a1);
		a3 = _mm_add_ps(a3, a1);
		a4 = _mm_add_ps(a4, a1);
		a5 = _mm_add_ps(a5, a1);
		
		//now join streams by adding previous three elements to each element

		//shift right by one element and add
		s0 = _mm_alignr_ps(a0, z, 12);
		s1 = _mm_alignr_ps(a1, a0, 12);
		s2 = _mm_alignr_ps(a2, a1, 12);
		s3 = _mm_alignr_ps(a3, a2, 12);
		s4 = _mm_alignr_ps(a4, a3, 12);
		s5 = _mm_alignr_ps(a5, a4, 12);
		
		a0 = _mm_add_ps(a0, s0);
		a1 = _mm_add_ps(a1, s1);
		a2 = _mm_add_ps(a2, s2);
		a3 = _mm_add_ps(a3, s3);
		a4 = _mm_add_ps(a4, s4);
		a5 = _mm_add_ps(a5, s5);

		//shift right by two elements and add
		s0 = _mm_alignr_ps(a0, z, 8);
		s1 = _mm_alignr_ps(a1, a0, 8);
		s2 = _mm_alignr_ps(a2, a1, 8);
		s3 = _mm_alignr_ps(a3, a2, 8);
		s4 = _mm_alignr_ps(a4, a3, 8);
		s5 = _mm_alignr_ps(a5, a4, 8);
		
		a0 = _mm_add_ps(a0, s0);
		a1 = _mm_add_ps(a1, s1);
		a2 = _mm_add_ps(a2, s2);
		a3 = _mm_add_ps(a3, s3);
		a4 = _mm_add_ps(a4, s4);
		a5 = _mm_add_ps(a5, s5);
		
		//reextract running total
		t = _mm_insert_ps(t, a5, _MM_SHUFFLE(3,0,0,14));

		_mm_store_ps(a, a0);
		_mm_store_ps(a + 4, a1);
		_mm_store_ps(a + 8, a2);
		_mm_store_ps(a + 12, a3);
		_mm_store_ps(a + 16, a4);
		_mm_store_ps(a + 20, a5);
		a += 24;
		n -= 24;
	}
	while(n > 0)
	{
		a0 = _mm_load_ss(a);
		t = _mm_add_ss(t, a0);
		_mm_store_ss(a, t);
		a++;
		n--;
	}
}

/*
	Buffers that we reuse with every test.
*/
float *buffer0 = NULL, *buffer1 = NULL, *buffer2 = NULL;
int bufferSize = 0;

uint64_t rdtsc(){
	/*
		return the relative number of clock ticks we've enjoyed since some epoc
	*/
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

int cmp_float_magnitude(const void *lp, const void *rp)
{
	//comparison function used for sorting floats
	float l = fabsf(*(float *)lp), r = fabsf(*(float *)rp);
	return l < r ? -1 : (l > r ? 1 : 0);
}
float sum(float *b, int n)
{
	//We need a gold standard to check the accuracy of our total sum against.
	//To do this we sort all the numbers being summed and add to smallest
	//magnitude ones first.
	qsort(b, n, sizeof(float), cmp_float_magnitude);
	float total = 0.0;
	for(int i = 0; i < n; i++)
	{
		total += b[i];
	}
	return total;
}

/*
	We return many values after testing one cumulative sum. We do this in 
	globals.
*/
double dt_result;
uint64_t dc_result;
double err_result;
double sum_err_result;

void timecumsum(int n, void (*f)(float *, int))
{
	/*
		This function tests one in-place cumulative sum function on one random
		array of floats of size approximately n.
	*/
	if(bufferSize + 96 < n || !buffer0 || !buffer1)
	{
		if(buffer0) free(buffer0);
		if(buffer1) free(buffer1);
		if(buffer2) free(buffer2);
		bufferSize = n + 96;
		buffer0 = malloc(sizeof(float) * bufferSize);
		buffer1 = malloc(sizeof(float) * bufferSize);
		buffer2 = malloc(sizeof(float) * bufferSize);
	}

	//We vary sizes and offsets randomly to make sure no algorithm enjoys an
	//unfair advantage of alignments.
	n += rand() % 16;

	int o = rand() % (bufferSize - n);
	float *b0 = buffer0 + o, *b1 = buffer1 + o;

	float scale = 100.0;
	for(int i = 0; i < n; i++)
	{
		b0[i] = scale * ((float)( 1000 * (rand() % 1000) + (rand() % 1000)) / 1000000.0 - 0.5);
		//Because our denominator is not a power of 2, these numbers can not
		//be accurately reflected in a float. This is necessary to test how
		//badly our algorithms ruin numerical stability. We could be more 
		//rigorous here.
	}
	//Compute a niave result to compare against as well as a copy we can sort
	//to check the total.
	float a, s = 0.0;
	for(int i = 0; i < n; i++)
	{
		a = b0[i];
		b1[i] = a + s;
		buffer2[i] = a;
		s += a;
	}
	
	//Do the test run and see how many cycles and how much time it took.
	struct timeval t;
	gettimeofday(&t, NULL);
	double t0 = (double)t.tv_sec + (double)t.tv_usec / +1e9;
	uint64_t c0 = rdtsc();
	f(b0, n);
	uint64_t c1 = rdtsc();
	gettimeofday(&t, NULL);
	double t1 = (double)t.tv_sec + (double)t.tv_usec / +1e9;
	
	//measure the per element error against the niave result.
	float e = 0.0;
	for(int i = 0; i < n; i++)
	{
		a = b0[i] - b1[i];
		a = a * a;
		e += a;
	}

	dt_result = t1 - t0;
	dc_result = c1 - c0;
	err_result = e / (double)n;

	//measure the error of the final total against a better than niave result.
	float stable_sum = sum(buffer2, n);
	a = b0[n-1] - stable_sum;
	sum_err_result = a * a;
}

/*
Various functions to confirm that instrinsic functions do what I think they do.
*/
void test_mm_alignr_ps()
{
	printf("\nTest _mm_alignr_ps\n");
	__m128 a, b, c;
	float _fa[8], _fb[8], _fc[8];
	float *fa = (float *)(((uint64_t)_fa + 15) & ~15ULL);
	float *fb = (float *)(((uint64_t)_fb + 15) & ~15ULL);
	float *fc = (float *)(((uint64_t)_fc + 15) & ~15ULL);
	a = _mm_set_ps(0.4f, 0.3f, 0.2f, 0.1f);
	b = _mm_set_ps(0.8f, 0.7f, 0.6f, 0.5f);
	c = _mm_alignr_ps(a, b, 8);
	float a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3;
	_mm_store_ps(fa, a);
	_mm_store_ps(fb, b);
	_mm_store_ps(fc, c);
	printf("a: %f, %f, %f, %f\n", fa[0], fa[1], fa[2], fa[3]);
	printf("b: %f, %f, %f, %f\n", fb[0], fb[1], fb[2], fb[3]);
	printf("c: %f, %f, %f, %f\n", fc[0], fc[1], fc[2], fc[3]);

}
void test_hadd()
{
	printf("\nTest _mm_hadd_ps\n");
	__m128 a, b, c;
	float _fa[8], _fb[8], _fc[8];
	float *fa = (float *)(((uint64_t)_fa + 15) & ~15ULL);
	float *fb = (float *)(((uint64_t)_fb + 15) & ~15ULL);
	float *fc = (float *)(((uint64_t)_fc + 15) & ~15ULL);
	a = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
	b = _mm_set_ps(8.0f, 7.0f, 6.0f, 5.0f);
	c = _mm_hadd_ps(a, b);
	float a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3;
	_mm_store_ps(fa, a);
	_mm_store_ps(fb, b);
	_mm_store_ps(fc, c);
	printf("a: %f, %f, %f, %f\n", fa[0], fa[1], fa[2], fa[3]);
	printf("b: %f, %f, %f, %f\n", fb[0], fb[1], fb[2], fb[3]);
	printf("c: %f, %f, %f, %f\n", fc[0], fc[1], fc[2], fc[3]);
}
void test_insert()
{
	printf("\nTest _mm_insert_ps\n");
	__m128 a, b, c;
	float _fa[8], _fb[8], _fc[8];
	float *fa = (float *)(((uint64_t)_fa + 15) & ~15ULL);
	float *fb = (float *)(((uint64_t)_fb + 15) & ~15ULL);
	float *fc = (float *)(((uint64_t)_fc + 15) & ~15ULL);
	a = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
	b = _mm_set_ps(8.0f, 7.0f, 6.0f, 5.0f);
	c = _mm_insert_ps(a, b, _MM_SHUFFLE(2, 1, 0, 0));
	_mm_store_ps(fa, a);
	_mm_store_ps(fb, b);
	_mm_store_ps(fc, c);
	printf("a: %f, %f, %f, %f\n", fa[0], fa[1], fa[2], fa[3]);
	printf("b: %f, %f, %f, %f\n", fb[0], fb[1], fb[2], fb[3]);
	printf("c: %f, %f, %f, %f\n", fc[0], fc[1], fc[2], fc[3]);
}
void test_shuffle()
{
	printf("\nTest _mm_shuffle_ps\n");
	__m128 a, b, c;
	float _fa[8], _fb[8], _fc[8];
	float *fa = (float *)(((uint64_t)_fa + 15) & ~15ULL);
	float *fb = (float *)(((uint64_t)_fb + 15) & ~15ULL);
	float *fc = (float *)(((uint64_t)_fc + 15) & ~15ULL);
	a = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
	b = _mm_set_ps(8.0f, 7.0f, 6.0f, 5.0f);
	c = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 1, 0));
	_mm_store_ps(fa, a);
	_mm_store_ps(fb, b);
	_mm_store_ps(fc, c);
	printf("a: %f, %f, %f, %f\n", fa[0], fa[1], fa[2], fa[3]);
	printf("b: %f, %f, %f, %f\n", fb[0], fb[1], fb[2], fb[3]);
	printf("c: %f, %f, %f, %f\n", fc[0], fc[1], fc[2], fc[3]);
}

//How many cumsum functions are we testing
#define N_SLOTS 3

int main(char **argv, int argc)
{
	test_hadd();
	test_insert();
	test_shuffle();
	test_mm_alignr_ps();

	int64_t totalsize = 0, n = 0;
	printf("\nBenchmarking cumulative sums.\n");
	struct {
		double totalTime, averageRate, averageError, averageSumError;
		uint64_t totalCycles;
		void (*f)(float *, int);
		char* name;
	} slots[N_SLOTS];
	int n_slots = N_SLOTS, i, order[N_SLOTS];

	for(i = 0; i < n_slots; i++) 
	{
		slots[i].totalTime = 0.0;
		slots[i].averageRate = 0.0;
		slots[i].averageError = 0.0;
		slots[i].totalCycles = 0;
		order[i] = i;
	}
	slots[0].f = cumsum_scaler;
	slots[0].name = "niave";
	slots[1].f = cumsum_super_scaler;
	slots[1].name = "super scalar";
	slots[2].f = cumsum_super_scaler_simd;
	slots[2].name = "super scalar simd";

	for(int64_t vsize = (1 << 18); vsize >= 1; vsize >>= 1)
	{

		int n_iterations = 100;

		printf("\nVector length: %ld\n", vsize);

		//prime cache
		for(i = 0; i < n_slots; i++)
		{
			timecumsum(vsize, slots[i].f);
			slots[i].totalTime = 0.0;
			slots[i].averageRate = 0.0;
			slots[i].averageError = 0.0;
			slots[i].averageSumError = 0.0;
			slots[i].totalCycles = 0;
		}

		for(int iterations = 0; iterations < n_iterations; iterations++)
		{
			//shuffle order so no one gets cache advantage
			for(i = 0; i < n_slots; i++)
			{
				int j = rand() % n_slots;
				int tmp = order[i];
				order[i] = order[j];
				order[j] = tmp;
			}

			for(i = 0; i < n_slots; i++)
			{
				int slot_i = order[i];
				timecumsum(vsize, slots[slot_i].f);
				slots[slot_i].totalTime += dt_result;
				slots[slot_i].totalCycles += dc_result;
				slots[slot_i].averageRate += (double)vsize / (double)dc_result;
				slots[slot_i].averageError += err_result;
				slots[slot_i].averageSumError += sum_err_result;
			}
		}
		for(i = 0; i < n_slots; i++)
		{
			slots[i].totalCycles /= n_iterations;
			slots[i].averageRate /= (double)n_iterations;
			slots[i].averageError /= (double)n_iterations;
			slots[i].averageSumError /= (double)n_iterations;
			printf("%17s: %12ld cycles per iteration, %10f elements per cycle, %10f avg error, %10f avg sum error\n", 
					slots[i].name, slots[i].totalCycles, slots[i].averageRate, slots[i].averageError, slots[i].averageSumError);
		}
	}

	//free the buffers allocated by timecumsum()
	free(buffer0);
	free(buffer1);

}





