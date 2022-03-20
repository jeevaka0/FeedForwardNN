/* Created by Jeevaka Dassanayake on 4/24/20.
   Copyright (c) 2020 Jeevaka Dassanayake. All rights reserved. */

#include "../../Headers/BaseIncludes.hpp"
#include "../../Headers/StdPlus.hpp"
#include "../../Headers/PlusPlus.hpp"

#include "Bases/WeightStrategy.hpp"
#include "Bases/ActivationStrategy.hpp"

#include "Bases/Layer.hpp"
#include "Bases/HiddenLayer.hpp"
#include "DenseLayer.hpp"


DenseLayer::DenseLayer( unsigned long I, double lrMultiplier, unsigned long _J, ActivationStrategy* pActivationStrategy
		, WeightStrategy* pWeightStrategy )
		: HiddenLayer( I, pActivationStrategy, pWeightStrategy ), J( _J ), lrMultiplier( lrMultiplier ) {
	W = new ( Align32 ) double[I * J.high() ];		// Each row has 'Jp' extra doubles.
	B = new ( Align32 ) double[I];
}


DenseLayer::~DenseLayer() {
	delete[] B;
	delete[] W;
}


void DenseLayer::initializeWeights() {
	pWeightStrategy->reset();
	pWeightStrategy->initialize( W, I, J );
}


void DenseLayer::initializeBias( double bias ) {
	double* b = B;
	for ( unsigned long i = 0; i < I; i++ ) {
		*b = bias;
		b++;
	}
}


void DenseLayer::testSummation() {
	unsigned long remainder = I % 4;
	double* pEnd = B + I - remainder;

	__m256i mask = LoadMask256[ remainder ];
	__m256d sum4 = _mm256_maskload_pd( pEnd, mask );

	__m256d* pArray = (__m256d*)B;		// B was allocated aligned.
	while ( pArray != (__m256d*)pEnd ) {
		sum4 = _mm256_add_pd( sum4, *pArray++ );
	}

	sum4 = _mm256_hadd_pd( sum4, sum4 );				// a,b,c,d => a+b,a+b,c+d,c+d
	__m128d sumHi = _mm256_extractf128_pd( sum4, 1 );	// a+b,a+b
	Z[0] = sum4[0] + sumHi[0];
}

/*
void DenseLayer::testSummation() { //noexcept {
	double sum = (J % 2) == 1 ? B[J-1] : 0;
	unsigned long j = 2*(J/2);
	do {
		sum += *(B+j-1);
		j -= 2;
		sum += *(B+j);
	} while ( j != 0 );
	Z[0] = sum;
}
*/

/*
void DenseLayer::testSummation() {
	double sum = 0;
	unsigned long j = 0;
	do {
		sum += *(B+j);
		j++;
	} while ( j != J.value() );
	Z[0] = sum;
	//cout << sum << endl;
}
*/

/*
void DenseLayer::testSummation() {
	double sum = 0;
	for( unsigned long i = 0; i < I; i++ ) {
		sum += *(B+i);
	}
	Z[0] = sum;
}
*/

/*
void DenseLayer::testSummation() {
	const double* b = B;
	Z[0] = 0;
	for( unsigned long i = 0; i < I; i++ ) {
		Z[0] += *b;
		b++;
	}
}
*/

#ifdef USE_AVX

void DenseLayer::doSummation( Layer* pPrevious ) {		// Eq. 1.1.
	__m256i mask = LoadMask256[ J.remainder() ];
	const __m256d* pA = (const __m256d*)pPrevious->getA();		// A was allocated aligned.
	__m256d* pW = (__m256d*)W;

	const double* b = B;
	unsigned long i = 0;
	__m256d aTail = _mm256_maskload_pd( (double*)pA + J.low(), mask );
	do {
		// Parts of A and W that do not fit into a 4-double block.
		__m256d w4 = _mm256_maskload_pd( (double*)pW + J.low(), mask );
		__m256d sum4 = _mm256_mul_pd( aTail, w4 );

		for( unsigned long j = 0; j < J.blocks(); j++ ) {
			__m256d a4 = _mm256_load_pd( (const double*)( pA + j ) );
			sum4 += _mm256_mul_pd( a4, *( pW + j ) );
		}

		sum4 = _mm256_hadd_pd( sum4, sum4 );				// a,b,c,d => a+b,a+b,c+d,c+d
		__m128d sumHi = _mm256_extractf128_pd( sum4, 1 );	// a+b,a+b
		Z[i] = sum4[0] + sumHi[0] + *b;

		b++;
		pW += J.high() / 4;
	} while ( ++i < I );
}


// Precondition. Current iteration deltas are updated.
// pPrevious: the layer that comes before current one in the forward pass order.
void DenseLayer::updateWeights( const Layer* pPrevious, double networkLearningRate ) {
	double learningRate = lrMultiplier * networkLearningRate;
	// Eq. 1.32 implementation.
	__m256d* pW = (__m256d*)W;

	double* b = B;
	const double* di = D;

	__m256i mask = LoadMask256[ J.remainder() ];
	__m256d aTail = _mm256_maskload_pd( pPrevious->getA() + J.low(), mask );
	__m256d* pEndA = (__m256d*)pPrevious->getA() + J.blocks();

	unsigned long i = 0;
	do {
		__m256d* pA = (__m256d*)pPrevious->getA();

		double ld = learningRate * *di;
		__m256d ld4 { ld, ld, ld, ld };

		while( pA < pEndA ) {
			__m256d p = _mm256_mul_pd( ld4, *pA++ );
			__m256d w4 = _mm256_add_pd( p, *pW );
			_mm256_store_pd( (double*)pW, w4 );
			pW++;
		}
		if ( J.padding() != 0 ) {
			__m256d p = _mm256_mul_pd( ld4, aTail );
			__m256d w4 = _mm256_add_pd( p, *pW );
			_mm256_store_pd( (double*)pW, w4 );
			pW++;
		}

		*b += ld;			// Bias term.
		b++;
		di++;							// Delta is per neuron. Hence increment once per neuron.
	} while ( ++i < I );
}

#else


void DenseLayer::doSummation( Layer* pPrevious ) {		// Eq. 1.1.
	const double* AJ = pPrevious->getA();
	const double* w = W;
	const double* b = B;
	for( unsigned long i = 0; i < I; i++ ) {
		const double* aj = AJ;
		double sum = 0;
		for( unsigned long j = 0; j < J.value(); j++ ) {		// Can we use vector arithmetic for this?
			sum += *aj * *w;
			aj++;
			w++;
		}
		Z[i] = sum + *b;		// The bias.
		b++;
		w += J.padding();
	}
}

// Precondition. Current iteration deltas are updated.
// pPrevious: the layer that comes before current one in the forward pass order.
void DenseLayer::updateWeights( const Layer* pPrevious, double networkLearningRate ) {
	double learningRate = lrMultiplier * networkLearningRate;
	// Eq. 1.24 implementation.
	const double* AJ = pPrevious->getA();			// A vector of length J.
	double* w = W;
	double* b = B;
	const double* di = D;

	for( unsigned long i = 0; i < I; i++ ) {
		const double* aj = AJ;
		for( unsigned long j = 0; j < J.value(); j++ ) {		// Can we use vector arithmetic for this?
			*w += learningRate * *di * *aj;
			w++;
			aj++;
		}
		*b += learningRate * *di;			// Bias term.
		b++;
		di++;							// Delta is per neuron. Hence increment once per neuron.
		w += J.padding();				// Move to next neuron.
	}
}

#endif


void DenseLayer::updateDelta( const double* sourceWeights, const double* sourceDeltas, unsigned long sourceWidth
		, const PackedN& sourceJ ) {
	pActivationStrategy->updateDelta( sourceWeights, sourceDeltas, sourceWidth, A, D, sourceJ );
}


const double* DenseLayer::getWeights() const {
	return W;
}


const PackedN& DenseLayer::getJ() const {
	return J;
}


// Does not include the bias term.
void DenseLayer::writeWeights( ostream& os, unsigned long i, unsigned long lineLength ) const {
	double* pW = W + i * J.high();
	CsvStream::PrintArray( os, pW, J.value(), lineLength );
}



// Neuron, Input, Weight
void DenseLayer::writeAllWeights( ostream& os, const string& lead ) const {
	double* pW = W;
	double* b = B;
	for ( unsigned long i = 0; i < I; i++ ) {
		os << lead;
		CsvStream::Print( os, i, 0, *b++ ) << endl;					// 0 index for bias term.
		for ( unsigned long j = 1; j <= J.value(); j++ ) {			// weights have indices 1 to J.
			os << lead;
			CsvStream::Print( os, i, j, *pW ) << endl;
			pW++;
		}
		pW += J.padding();
	}
}


void DenseLayer::writeAllZAD( ostream& os, const string& lead, const double* pT ) const {
	for ( unsigned long i = 0; i < I; i++ ) {
		os << lead;
		CsvStream::Print( os, i, Z[i], A[i], D[i] ) << CsvStream::Delimiter;
		if ( nullptr == pT ) {
			os << "NA";
		} else {
			os << *( pT + i );
		}
		os << endl;					// 0 index for bias term.
	}
}


double DenseLayer::getLearningRateMultiplier() const {
	return lrMultiplier;
}
