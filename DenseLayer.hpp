/* Created by Jeevaka Dassanayake on 4/24/20.
   Copyright (c) 2020 Jeevaka Dassanayake. All rights reserved. */

class DenseLayer : public HiddenLayer {
public :
	DenseLayer( unsigned long I, double lrMultiplier, unsigned long J, ActivationStrategy* pActivationStrategy
			, WeightStrategy* pWeightStrategy );
	virtual ~DenseLayer();
	void initializeWeights();
	void initializeBias( double bias );

	void doSummation( Layer* pPrevious ) override;
	void updateWeights( const Layer* pPrevious, double networkLearningRate ) override;			// Eq. 1.24
	void updateDelta( const double* sourceWeights, const double* sourceDeltas, unsigned long sourceWidth
			, const PackedN& sourceJ ) override;

	const double* getWeights() const;
	const PackedN& getJ() const;
	double getLearningRateMultiplier() const;
	void testSummation();// noexcept;

	void writeWeights( ostream& os, unsigned long i, unsigned long lineLength ) const override;
	void writeAllWeights( ostream& os, const string& lead ) const override;
	void writeAllZAD( ostream& os, const string& lead, const double* pT ) const override;

protected :
	// Typedefs
	friend class WeightStrategy;


	// Methods


	// Data
	const PackedN J;
	double lrMultiplier;
	double* W;		// W is a I * ( J + Jp ) array.
					// All the weights associated with neuron 0 comes first. I.e. in the first row.
	//typedef __attribute__(( aligned(32))) double aligned_double;
	//aligned_double* B;		// Bias vector.
	double* B;
};


#define USE_AVX
