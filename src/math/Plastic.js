/**
 * Plastic.js
 * Mathematical constants and functions based on the Plastic Ratio (rho).
 * Used for the Phillips Rendering System's low-discrepancy sampling.
 */

// The Plastic Constant (rho) is the unique real solution to x^3 = x + 1
// Precise value: 1.32471795724474602596090885447809...
export const PLASTIC_CONSTANT = 1.324717957244746;

/**
 * Generates the Padovan sequence up to n terms.
 * The sequence is defined by P(n) = P(n-2) + P(n-3) with P(0)=P(1)=P(2)=1.
 * @param {number} n - The number of terms to generate.
 * @returns {number[]} Array containing the Padovan sequence.
 */
export function getPadovanSequence(n) {
    if (n <= 0) return [];
    if (n === 1) return [1];
    if (n === 2) return [1, 1];
    if (n === 3) return [1, 1, 1];

    const sequence = [1, 1, 1];
    for (let i = 3; i < n; i++) {
        const nextVal = sequence[i - 2] + sequence[i - 3];
        sequence.push(nextVal);
    }
    return sequence;
}

/**
 * Generates a low-discrepancy sampling point based on the Plastic Ratio.
 * Used for distributing splat centers or sampling patterns.
 * Based on the R_d sequence generalization for d=2 using the Plastic Constant.
 *
 * @param {number} index - The index of the sample (0, 1, 2, ...).
 * @returns {Object} {x, y} coordinates in [0, 1) range.
 */
export function getPlasticSamplingPoint(index) {
    // 2D low-discrepancy sequence using the Plastic Constant (rho)
    // alpha = (1/rho, 1/rho^2) mod 1
    // or generalized as: phi_d = 1/rho^(d)

    const rho = PLASTIC_CONSTANT;
    const a1 = 1.0 / rho;
    const a2 = 1.0 / (rho * rho);

    return {
        x: (0.5 + a1 * index) % 1.0,
        y: (0.5 + a2 * index) % 1.0
    };
}
