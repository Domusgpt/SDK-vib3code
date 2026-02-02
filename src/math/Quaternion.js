/**
 * Quaternion.js
 * Core mathematical class for handling hyper-complex numbers in the Quatossian Framework.
 * Represents orientation "spin-states" for Gaussian Kernels.
 */

export class Quaternion {
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    constructor(x = 0, y = 0, z = 0, w = 1) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    /**
     * Set from axis-angle representation.
     * @param {Object} axis - {x, y, z} normalized axis.
     * @param {number} angle - Angle in radians.
     */
    setFromAxisAngle(axis, angle) {
        const halfAngle = angle / 2, s = Math.sin(halfAngle);
        this.x = axis.x * s;
        this.y = axis.y * s;
        this.z = axis.z * s;
        this.w = Math.cos(halfAngle);
        return this;
    }

    /**
     * Normalize the quaternion.
     * Essential for maintaining "coherence" in the spin state.
     */
    normalize() {
        let l = Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
        if (l === 0) {
            this.x = 0; this.y = 0; this.z = 0; this.w = 1;
        } else {
            l = 1 / l;
            this.x *= l;
            this.y *= l;
            this.z *= l;
            this.w *= l;
        }
        return this;
    }

    /**
     * Multiply this quaternion by another (Hamilton product).
     * Represents combining two rotations/spins.
     */
    multiply(q) {
        return this.multiplyQuaternions(this, q);
    }

    multiplyQuaternions(a, b) {
        const qax = a.x, qay = a.y, qaz = a.z, qaw = a.w;
        const qbx = b.x, qby = b.y, qbz = b.z, qbw = b.w;

        this.x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby;
        this.y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz;
        this.z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx;
        this.w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz;
        return this;
    }

    toArray() {
        return [this.x, this.y, this.z, this.w];
    }
}
