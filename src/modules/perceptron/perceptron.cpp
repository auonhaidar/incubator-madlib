/* ------------------------------------------------------
 *
 * @file perceptron.cpp
 *
 * @brief perceptron functions
 *
 *
 *//* ----------------------------------------------------------------------- */
#include <limits>
#include <dbconnector/dbconnector.hpp>
#include <modules/shared/HandleTraits.hpp>
#include <modules/prob/boost.hpp>
#include <boost/math/distributions.hpp>
#include <modules/prob/student.hpp>
#include "perceptron.hpp"

namespace madlib {

// Use Eigen
using namespace dbal::eigen_integration;

namespace modules {

// Import names from other MADlib modules
using dbal::NoSolutionFoundException;

namespace perceptron {

enum { IN_PROCESS, COMPLETED, TERMINATED, NULL_EMPTY };


AnyType stateToResult(const Allocator &inAllocator,
                      const HandleMap<const ColumnVector, TransparentHandle<double> >& inCoef,
                      const Matrix & hessian,
                      const double &logLikelihood,
                      int status,
                      const uint64_t &numRows);


/**
 * @brief SIGMOIDAL
 */
inline double sigma(double x) {
    return 1. / (1. + std::exp(-x));
}

/**
 * @brief TANH
 */
inline double tanh(double x) {
    return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
}

/**
 * @brief ReLU: softplus
 */
inline double relu(double x) {
    return std::log(1.0 + std::exp(x));
}



template <class Handle>
class PerceptronIGDTransitionState {
    template <class OtherHandle>
    friend class PerceptronIGDTransitionState;

  public:
    PerceptronIGDTransitionState(const AnyType &inArray)
        : mStorage(inArray.getAs<Handle>()) {

        rebind(static_cast<uint16_t>(mStorage[0]));
    }

    
    inline operator AnyType() const {
        return mStorage;
    }

    
    inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
        rebind(inWidthOfX);
        widthOfX = inWidthOfX;
    }

    
    template <class OtherHandle>
    PerceptronIGDTransitionState &operator=(
        const PerceptronIGDTransitionState<OtherHandle> &inOtherState) {

        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    
    template <class OtherHandle>
    PerceptronIGDTransitionState &operator+=(
        const PerceptronIGDTransitionState<OtherHandle> &inOtherState) {

        if (mStorage.size() != inOtherState.mStorage.size() ||
            widthOfX != inOtherState.widthOfX)
            throw std::logic_error("Internal error: Incompatible transition "
                                   "states");

        
        double totalNumRows = static_cast<double>(numRows)
            + static_cast<double>(inOtherState.numRows);
        coef = double(numRows) / totalNumRows * coef
            + double(inOtherState.numRows) / totalNumRows * inOtherState.coef;

        numRows += inOtherState.numRows;
        X_transp_AX += inOtherState.X_transp_AX;
        logLikelihood += inOtherState.logLikelihood;
        status = (inOtherState.status == TERMINATED) ? inOtherState.status : status;
        return *this;
    }

    
    inline void reset() {
        stepsize = .01;
        numRows = 0;
        X_transp_AX.fill(0);
        logLikelihood = 0;
        status = IN_PROCESS;
    }

  private:
    static inline uint32_t arraySize(const uint16_t inWidthOfX) {
        return 5 + inWidthOfX * inWidthOfX + inWidthOfX;
    }
    
    void rebind(uint16_t inWidthOfX) {
        widthOfX.rebind(&mStorage[0]);
        stepsize.rebind(&mStorage[1]);
        coef.rebind(&mStorage[2], inWidthOfX);
        numRows.rebind(&mStorage[2 + inWidthOfX]);
        X_transp_AX.rebind(&mStorage[3 + inWidthOfX], inWidthOfX, inWidthOfX);
        logLikelihood.rebind(&mStorage[3 + inWidthOfX * inWidthOfX + inWidthOfX]);
        status.rebind(&mStorage[4 + inWidthOfX * inWidthOfX + inWidthOfX]);
    }

    Handle mStorage;

  public:
    typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
    typename HandleTraits<Handle>::ReferenceToDouble stepsize;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;

    typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap X_transp_AX;
    typename HandleTraits<Handle>::ReferenceToDouble logLikelihood;
    typename HandleTraits<Handle>::ReferenceToUInt16 status;
};

AnyType
perceptron_igd_step_transition::run(AnyType &args) {
    PerceptronIGDTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }
    double y = args[1].getAs<bool>() ? 1. : -1.;
    MappedColumnVector x;
    try {
        // an exception is raised in the backend if args[2] contains nulls
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        // x is a const reference, we can only rebind to change its pointer
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }
    if (!x.is_finite()){
        warning("Design matrix is not finite.");
        state.status = TERMINATED;
        return state;
    }

    // We only know the number of independent variables after seeing the first
    // row.
    if (state.numRows == 0) {
        if (x.size() > std::numeric_limits<uint16_t>::max()){
            warning("Number of independent variables cannot be larger than 65535.");
            state.status = TERMINATED;
            return state;
        }

        state.initialize(*this, static_cast<uint16_t>(x.size()));

        // For the first iteration, the previous state is NULL
        if (!args[3].isNull()) {
            PerceptronIGDTransitionState<ArrayHandle<double> > previousState = args[3];
            state = previousState;
            state.reset();
        }
    }

    // Now do the transition step
    state.numRows++;

    // xc = x^T_i c
    double xc = dot(x, state.coef);
    double scale = state.stepsize * sigma(-xc * y) * y;
    state.coef += scale * x;

    // Note: previous coefficients are used for Hessian and log likelihood
    if (!args[3].isNull()) {
        PerceptronIGDTransitionState<ArrayHandle<double> > previousState = args[3];

        double previous_xc = dot(x, previousState.coef);

        // a_i = sigma(x_i c) sigma(-x_i c)
        double a = sigma(previous_xc) * sigma(-previous_xc);
        //triangularView<Lower>(state.X_transp_AX) += x * trans(x) * a;
        state.X_transp_AX += x * trans(x) * a;

        // l_i(c) = - ln(1 + exp(-y_i * c^T x_i))
        state.logLikelihood -= std::log( 1. + std::exp(-y * previous_xc) );
    }

    return state;
}

AnyType
perceptron_igd_step_merge_states::run(AnyType &args) {
    PerceptronIGDTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    PerceptronIGDTransitionState<ArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    // Merge states together and return
    stateLeft += stateRight;
    return stateLeft;
}


AnyType
perceptron_igd_step_final::run(AnyType &args) {
    PerceptronIGDTransitionState<MutableArrayHandle<double> > state = args[0];

    if(!state.coef.is_finite()){
        warning("Overflow or underflow in incremental-gradient iteration. Input"
              "data is likely of poor numerical condition.");
        state.status = TERMINATED;
        return state;
    }

    // Aggregates that haven't seen any data just return Null.
    if (state.numRows == 0){
        state.status = NULL_EMPTY;
        return state;
    }

    return state;
}

/**
 * @brief Return the difference in log-likelihood between two states
 */
AnyType
internal_perceptron_igd_step_distance::run(AnyType &args) {
    PerceptronIGDTransitionState<ArrayHandle<double> > stateLeft = args[0];
    PerceptronIGDTransitionState<ArrayHandle<double> > stateRight = args[1];

    if(stateLeft.status == NULL_EMPTY || stateRight.status == NULL_EMPTY){
        return 0.0;
    }

    return std::abs(stateLeft.logLikelihood - stateRight.logLikelihood);
}

/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType
internal_perceptron_igd_result::run(AnyType &args) {
    PerceptronIGDTransitionState<ArrayHandle<double> > state = args[0];

    if (state.status == NULL_EMPTY)
        return Null();

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);

    return stateToResult(*this, state.coef,
                         state.X_transp_AX,
                         state.logLikelihood,
                         state.status, state.numRows);
}


AnyType stateToResult(
    const Allocator &inAllocator,
    const HandleMap<const ColumnVector, TransparentHandle<double> > &inCoef,
    const Matrix & hessian,
    const double &logLikelihood,
    int status,
    const uint64_t &numRows) {

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        hessian, EigenvaluesOnly, ComputePseudoInverse);

    const Matrix &inverse_of_X_transp_AX = decomposition.pseudoInverse();
    const ColumnVector &diagonal_of_X_transp_AX = inverse_of_X_transp_AX.diagonal();

    MutableNativeColumnVector stdErr(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector waldZStats(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector waldPValues(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector oddsRatios(
        inAllocator.allocateArray<double>(inCoef.size()));

    for (Index i = 0; i < inCoef.size(); ++i) {
        stdErr(i) = std::sqrt(diagonal_of_X_transp_AX(i));
        waldZStats(i) = inCoef(i) / stdErr(i);
        waldPValues(i) = 2. * prob::cdf( prob::normal(),
                                         -std::abs(waldZStats(i)));
        oddsRatios(i) = std::exp( inCoef(i) );
    }

    
    AnyType tuple;
    tuple << inCoef << logLikelihood << stdErr << waldZStats << waldPValues
          << oddsRatios << inverse_of_X_transp_AX
          << sqrt(decomposition.conditionNo()) << status << numRows;
    return tuple;
}


template <class Handle>
class RobustPerceptronTransitionState {
    template <class OtherHandle>
    friend class RobustPerceptronTransitionState;

  public:
    RobustPerceptronTransitionState(const AnyType &inArray)
        : mStorage(inArray.getAs<Handle>()) {

        rebind(static_cast<uint16_t>(mStorage[1]));
    }

    
    inline operator AnyType() const {
        return mStorage;
    }

    
    inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
        rebind(inWidthOfX);
        widthOfX = inWidthOfX;
    }

    
    template <class OtherHandle>
    RobustPerceptronTransitionState &operator=(
        const RobustPerceptronTransitionState<OtherHandle> &inOtherState) {

        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    
    template <class OtherHandle>
    RobustPerceptronTransitionState &operator+=(
        const RobustPerceptronTransitionState<OtherHandle> &inOtherState) {

        if (mStorage.size() != inOtherState.mStorage.size() ||
            widthOfX != inOtherState.widthOfX)
            throw std::logic_error("Internal error: Incompatible transition "
                                   "states");

        numRows += inOtherState.numRows;
        X_transp_AX += inOtherState.X_transp_AX;
        meat += inOtherState.meat;
        return *this;
    }

    
    inline void reset() {
        numRows = 0;
        X_transp_AX.fill(0);
        meat.fill(0);

    }

  private:
    static inline size_t arraySize(const uint16_t inWidthOfX) {
        return 4 + 2 * inWidthOfX * inWidthOfX + inWidthOfX;
    }

    
    void rebind(uint16_t inWidthOfX) {
        iteration.rebind(&mStorage[0]);
        widthOfX.rebind(&mStorage[1]);
        coef.rebind(&mStorage[2], inWidthOfX);
        numRows.rebind(&mStorage[2 + inWidthOfX]);
        X_transp_AX.rebind(&mStorage[3 + inWidthOfX], inWidthOfX, inWidthOfX);
        meat.rebind(&mStorage[3 + inWidthOfX * inWidthOfX + inWidthOfX], inWidthOfX, inWidthOfX);
    }

    Handle mStorage;

  public:
    typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
    typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;

    typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap X_transp_AX;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap meat;
};



AnyType robuststateToResult(
    const Allocator &inAllocator,
    const ColumnVector &inCoef,
    const ColumnVector &diagonal_of_varianceMat) {

    MutableNativeColumnVector variance(
        inAllocator.allocateArray<double>(inCoef.size()));

    MutableNativeColumnVector coef(
        inAllocator.allocateArray<double>(inCoef.size()));

    MutableNativeColumnVector stdErr(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector waldZStats(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector waldPValues(
        inAllocator.allocateArray<double>(inCoef.size()));

    for (Index i = 0; i < inCoef.size(); ++i) {
        //variance(i) = diagonal_of_varianceMat(i);
        coef(i) = inCoef(i);

        stdErr(i) = std::sqrt(diagonal_of_varianceMat(i));
        waldZStats(i) = inCoef(i) / stdErr(i);
        waldPValues(i) = 2. * prob::cdf(
            prob::normal(), -std::abs(waldZStats(i)));
    }

    AnyType tuple;
    tuple <<  coef<<stdErr << waldZStats << waldPValues;
    return tuple;
}


AnyType
robust_perceptron_step_transition::run(AnyType &args) {
    if(args[0].isNull())
        return Null();
    RobustPerceptronTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }
    double y = args[1].getAs<bool>() ? 1. : -1.;
    MappedColumnVector x;
    try {
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }
    MappedColumnVector coef = args[3].getAs<MappedColumnVector>();

    if (!dbal::eigen_integration::isfinite(x)) {
        warning("Design matrix is not finite.");
        return Null();
    }

    if (state.numRows == 0) {
        if (x.size() > std::numeric_limits<uint16_t>::max()) {
            
            warning("Number of independent variables cannot be larger than 65535.");
            return Null();
        }

        state.initialize(*this, static_cast<uint16_t>(x.size()));
        state.coef = coef; //Copy this into the state for later
    }

    // Now do the transition step
    state.numRows++;
    double xc = dot(x, coef);
    ColumnVector Grad;
    Grad = sigma(-y * xc) * y * trans(x);

    Matrix GradGradTranspose;
    GradGradTranspose = Grad*Grad.transpose();
    state.meat += GradGradTranspose;

    // Note: sigma(-x) = 1 - sigma(x).
    // a_i = sigma(x_i c) sigma(-x_i c)
    double a = sigma(xc) * sigma(-xc);
    triangularView<Lower>(state.X_transp_AX) += x * trans(x) * a;
    return state;
}


AnyType
robust_perceptron_step_merge_states::run(AnyType &args) {
    if(args[0].isNull() || args[1].isNull())
        return Null();

    RobustPerceptronTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    RobustPerceptronTransitionState<ArrayHandle<double> > stateRight = args[1];
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    stateLeft += stateRight;
    return stateLeft;
}


AnyType
robust_perceptron_step_final::run(AnyType &args) {
    if (args[0].isNull())
        return Null();
    RobustPerceptronTransitionState<MutableArrayHandle<double> > state = args[0];
    if (state.numRows == 0)
        return Null();

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);

    Matrix bread = decomposition.pseudoInverse();

    
    Matrix varianceMat;// = meat;
    varianceMat = bread*state.meat*bread;

    

    return robuststateToResult(*this, state.coef,
                               varianceMat.diagonal());
}


template <class Handle>
class MarginalPerceptronTransitionState {
    template <class OtherHandle>
    friend class MarginalPerceptronTransitionState;

  public:
    MarginalPerceptronTransitionState(const AnyType &inArray)
        : mStorage(inArray.getAs<Handle>()) {

        rebind(static_cast<uint16_t>(mStorage[1]));
    }

    
    inline operator AnyType() const {
        return mStorage;
    }

    
    inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
        rebind(inWidthOfX);
        widthOfX = inWidthOfX;
    }

    
    template <class OtherHandle>
    MarginalPerceptronTransitionState &operator=(
        const MarginalPerceptronTransitionState<OtherHandle> &inOtherState) {

        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    
    template <class OtherHandle>
    MarginalPerceptronTransitionState &operator+=(
        const MarginalPerceptronTransitionState<OtherHandle> &inOtherState) {

        if (mStorage.size() != inOtherState.mStorage.size() ||
            widthOfX != inOtherState.widthOfX)
            throw std::logic_error("Internal error: Incompatible transition "
                                   "states");

        numRows += inOtherState.numRows;
        marginal_effects_per_observation += inOtherState.marginal_effects_per_observation;
        X_bar += inOtherState.X_bar;
        X_transp_AX += inOtherState.X_transp_AX;
        delta += inOtherState.delta;
        return *this;
    }

   
    inline void reset() {
        numRows = 0;
        marginal_effects_per_observation = 0;
        X_bar.fill(0);
        X_transp_AX.fill(0);
        delta.fill(0);
    }

  private:
    static inline size_t arraySize(const uint16_t inWidthOfX) {
        return 4 + 2 * inWidthOfX * inWidthOfX + 2 * inWidthOfX;
    }

    
    void rebind(uint16_t inWidthOfX) {
        iteration.rebind(&mStorage[0]);
        widthOfX.rebind(&mStorage[1]);
        coef.rebind(&mStorage[2], inWidthOfX);
        numRows.rebind(&mStorage[2 + inWidthOfX]);
        marginal_effects_per_observation.rebind(&mStorage[3 + inWidthOfX]);
        X_bar.rebind(&mStorage[4 + inWidthOfX], inWidthOfX);
        X_transp_AX.rebind(&mStorage[4 + 2*inWidthOfX], inWidthOfX, inWidthOfX);
        delta.rebind(&mStorage[4+inWidthOfX*inWidthOfX+2*inWidthOfX], inWidthOfX, inWidthOfX);
    }
    Handle mStorage;

  public:

    typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
    typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;
    typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
    typename HandleTraits<Handle>::ReferenceToDouble marginal_effects_per_observation;

    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap X_bar;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap X_transp_AX;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap delta;
};



AnyType marginalstateToResult(
    const Allocator &inAllocator,
    const ColumnVector &inCoef,
    const ColumnVector &diagonal_of_variance_matrix,
    const double inmarginal_effects_per_observation,
    const double numRows) {

    MutableNativeColumnVector marginal_effects(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector coef(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector stdErr(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector tStats(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector pValues(
        inAllocator.allocateArray<double>(inCoef.size()));

    for (Index i = 0; i < inCoef.size(); ++i) {
        coef(i) = inCoef(i);
        marginal_effects(i) = inCoef(i) * inmarginal_effects_per_observation / numRows;
        stdErr(i) = std::sqrt(diagonal_of_variance_matrix(i));
        tStats(i) = marginal_effects(i) / stdErr(i);

        if (numRows > inCoef.size())
            pValues(i) = 2. * prob::cdf(
                prob::normal(), -std::abs(tStats(i)));
    }

    AnyType tuple;
    tuple << marginal_effects
          << coef
          << stdErr
          << tStats
          << (numRows > inCoef.size()? pValues: Null());
    return tuple;
}


AnyType
marginal_perceptron_step_transition::run(AnyType &args) {
    if (args[0].isNull())
        return Null();
    MarginalPerceptronTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }
    MappedColumnVector x;
    try {
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }

    MappedColumnVector coef = args[3].getAs<MappedColumnVector>();
    if (!dbal::eigen_integration::isfinite(x)) {
        warning("Design matrix is not finite.");
        return Null();
    }

    if (state.numRows == 0) {
        if (x.size() > std::numeric_limits<uint16_t>::max()) {
            warning("Number of independent variables cannot be larger than 65535.");
            return Null();
        }
        state.initialize(*this, static_cast<uint16_t>(x.size()));
        state.coef = coef; //Copy this into the state for later
    }

    // Now do the transition step
    state.numRows++;
    double xc = dot(x, coef);
    double p = std::exp(xc)/ (1 + std::exp(xc));
    double a = sigma(xc) * sigma(-xc);

    // TODO: Change the average code so it won't overflow
    state.marginal_effects_per_observation += p * (1 - p);
    state.X_bar += x;
    state.X_transp_AX += x * trans(x) * a;

    Matrix delta;
    delta = (1 - 2*p) * state.coef * trans(x);
    // This should be faster than adding an identity
    for (int i=0; i < state.widthOfX; i++){
        delta(i,i) += 1;
    }

    // Standard error according to the delta method
    state.delta += p * (1 - p) * delta;

    return state;
}



AnyType
marginal_perceptron_step_merge_states::run(AnyType &args) {
    if(args[0].isNull() || args[1].isNull())
        return Null();

    MarginalPerceptronTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    MarginalPerceptronTransitionState<ArrayHandle<double> > stateRight = args[1];
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    stateLeft += stateRight;
    return stateLeft;
}

/**
 * @brief Marginal effects: Final step
 */
AnyType
marginal_perceptron_step_final::run(AnyType &args) {
    if (args[0].isNull())
        return Null();

    MarginalPerceptronTransitionState<MutableArrayHandle<double> > state = args[0];
    if (state.numRows == 0)
        return Null();

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);
    Matrix variance = decomposition.pseudoInverse();
    Matrix std_err;
    std_err = state.delta * variance * trans(state.delta) / static_cast<double>(state.numRows*state.numRows);

    return marginalstateToResult(*this,
                                 state.coef,
                                 std_err.diagonal(),
                                 state.marginal_effects_per_observation,
                                 static_cast<double>(state.numRows));
}



AnyType perceptron_predict::run(AnyType &args) {
    try {
        args[0].getAs<MappedColumnVector>();
    } catch (const ArrayWithNullException &e) {
        throw std::runtime_error(
            "Perceptron error: the coefficients contain NULL values");
    }

    try {
        args[1].getAs<MappedColumnVector>();
    } catch (const ArrayWithNullException &e) {
        return Null();
    }

    MappedColumnVector vec1 = args[0].getAs<MappedColumnVector>();
    MappedColumnVector vec2 = args[1].getAs<MappedColumnVector>();

    if (vec1.size() != vec2.size())
        throw std::runtime_error(
            "Coefficients and independent variables are of incompatible length");

    return vec1.dot(vec2) > 0 ? true : false;
}

AnyType perceptron_predict_prob::run(AnyType &args) {
    try {
        args[0].getAs<MappedColumnVector>();
    } catch (const ArrayWithNullException &e) {
        throw std::runtime_error(
            "Perceptron error: the coefficients contain NULL values");
    }

    try {
        args[1].getAs<MappedColumnVector>();
    } catch (const ArrayWithNullException &e) {
        return Null();
    }

    MappedColumnVector vec1 = args[0].getAs<MappedColumnVector>();
    MappedColumnVector vec2 = args[1].getAs<MappedColumnVector>();

    if (vec1.size() != vec2.size())
        throw std::runtime_error(
            "Coefficients and independent variables are of incompatible length");

    double dot = vec1.dot(vec2);
    double logit = 0.0;
    try {
        logit = 1.0 / (1 + exp(-dot));
    } catch (...) {
        logit = (dot > 0) ? 1.0 : 0.0;
    }
    return logit;
}

}
}
}
