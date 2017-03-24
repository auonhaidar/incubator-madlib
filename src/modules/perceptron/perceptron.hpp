/* ----------------------------------------------------------------------- *//**
 *
 * @file perceptron.hpp
 *
 *//* ----------------------------------------------------------------------- */


/**
 * @brief perceptron (incremetal-gradient step): Transition function
 */
DECLARE_UDF(perceptron, perceptron_igd_step_transition)

/**
 * @brief perceptron (incremetal-gradient step): State merge function
 */
DECLARE_UDF(perceptron, perceptron_igd_step_merge_states)

/**
 * @brief perceptron (incremetal-gradient step): Final function
 */
DECLARE_UDF(perceptron, perceptron_igd_step_final)

/**
 * @brief perceptron (incremetal-gradient step): Difference in
 *     log-likelihood between two transition states
 */
DECLARE_UDF(perceptron, internal_perceptron_igd_step_distance)

/**
 * @brief perceptron (iteratively-reweighted-lest-squares step):
 *     Convert transition state to result tuple
 */
DECLARE_UDF(perceptron, internal_perceptron_igd_result)

/**
 * @brief Robust Variance perceptron step: Transition function
 */
DECLARE_UDF(perceptron, robust_perceptron_step_transition)

/**
 * @brief Robust Variance perceptron step: State merge function
 */
DECLARE_UDF(perceptron, robust_perceptron_step_merge_states)

/**
 * @brief Robust Variance perceptron step: Final function
 */
DECLARE_UDF(perceptron, robust_perceptron_step_final)


/**
 * @brief Marginal Effects perceptron step: Transition function
 */
DECLARE_UDF(perceptron, marginal_perceptron_step_transition)

/**
 * @brief Marginal effects perceptron step: State merge function
 */
DECLARE_UDF(perceptron, marginal_perceptron_step_merge_states)

/**
 * @brief Marginal Effects perceptron step: Final function
 */
DECLARE_UDF(perceptron, marginal_perceptron_step_final)

/**
 * @brief Prediction functions
 */
DECLARE_UDF(perceptron, perceptron_predict)
DECLARE_UDF(perceptron, perceptron_predict_prob)
