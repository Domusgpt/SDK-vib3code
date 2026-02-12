/**
 * Quaternion.js
 * Quatossian Spin Algebra (4D Rotation)
 *
 * Provides robust quaternion operations for the E8 framework.
 * Focuses on unit quaternion operations used for 3D/4D rotation.
 */

export class Quaternion {
    /**
     * Creates a new Quaternion [x, y, z, w].
     * Defaults to Identity (0, 0, 0, 1).
     */
    constructor(x = 0, y = 0, z = 0, w = 1) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    /**
     * Multiplies two quaternions (Hamilton Product).
     * @param {Quaternion|Object} a
     * @param {Quaternion|Object} b
     * @returns {Quaternion} New quaternion resulting from a * b.
     */
    static multiply(a, b) {
        return new Quaternion(
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
        );
    }

    /**
     * Conjugates a quaternion (inverse of unit quaternion).
     * @param {Quaternion} q
     * @returns {Quaternion} New conjugate quaternion.
     */
    static conjugate(q) {
        return new Quaternion(-q.x, -q.y, -q.z, q.w);
    }

    /**
     * Rotates a 3D vector by a quaternion.
     * v' = q * v * q^-1
     * @param {Object} v - Vector {x, y, z}
     * @param {Quaternion} q - Rotation quaternion
     * @returns {Object} Rotated vector {x, y, z}
     */
    static rotateVector(v, q) {
        // Optimized implementation
        // t = 2 * cross(q.xyz, v)
        // v' = v + q.w * t + cross(q.xyz, t)

        const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
        const vx = v.x, vy = v.y, vz = v.z;

        // t = 2 * cross(q.xyz, v)
        const tx = 2 * (qy * vz - qz * vy);
        const ty = 2 * (qz * vx - qx * vz);
        const tz = 2 * (qx * vy - qy * vx);

        // v' = v + w*t + cross(q.xyz, t)
        return {
            x: vx + qw * tx + (qy * tz - qz * ty),
            y: vy + qw * ty + (qz * tx - qx * tz),
            z: vz + qw * tz + (qx * ty - qy * tx)
        };
    }

    /**
     * Rotates a 4D vector using left-isoclinic rotation (Moxness Folding).
     * v' = q * v (where v is treated as a quaternion)
     * @param {Float32Array} v4 - [x, y, z, w]
     * @param {Quaternion} q - Rotation quaternion
     * @returns {Float32Array} Rotated 4D vector
     */
    static rotate4DLeft(v4, q) {
        // q * v (Hamilton Product)
        // v4 is treated as a quaternion (x,y,z,w)
        const vx = v4[0], vy = v4[1], vz = v4[2], vw = v4[3];
        const qx = q.x, qy = q.y, qz = q.z, qw = q.w;

        return new Float32Array([
            qw * vx + qx * vw + qy * vz - qz * vy,
            qw * vy - qx * vz + qy * vw + qz * vx,
            qw * vz + qx * vy - qy * vx + qz * vw,
            qw * vw - qx * vx - qy * vy - qz * vz
        ]);
    }

    /**
     * Converts to Float32Array [x, y, z, w].
     */
    toFloat32Array() {
        return new Float32Array([this.x, this.y, this.z, this.w]);
    }

    /**
     * Creates a quaternion from an axis and an angle.
     * @param {Object} axis - Normalized {x, y, z}
     * @param {number} angle - Angle in radians
     * @returns {Quaternion}
     */
    static fromAxisAngle(axis, angle) {
        const halfAngle = angle * 0.5;
        const s = Math.sin(halfAngle);
        return new Quaternion(
            axis.x * s,
            axis.y * s,
            axis.z * s,
            Math.cos(halfAngle)
        );
    }
}
