/**
 * Perlin's improved gradient noise, in three dimensions.
 *
 * **Written here rather than installed.** It is sixty lines, it has no
 * configuration, and it is the same sixty lines it has been since 2002; a
 * dependency for it would be a `package.json` entry, a lockfile line and a
 * supply-chain surface in exchange for nothing.
 *
 * **In `lib` because two things now ride it**: the application background
 * (`app/BackgroundLayer`) and the landing page (`features/landing`). It was
 * written for the first and moved here for the second rather than copied —
 * a hundred and twenty lines of permutation table duplicated across two
 * features is a hundred and twenty lines that can silently disagree.
 *
 * The third dimension is time. Sampling `noise3(x, y, t)` rather than
 * `noise2(x, y)` is what makes a flow field *evolve* instead of being a fixed
 * pattern that particles slide along forever — with a static field, several
 * hundred particles find its attractor lines within a few seconds and the
 * drift collapses into a handful of permanent streaks.
 *
 * The permutation table is seeded from a constant, so the field is identical
 * on every load and in every test run. A random seed would make "why does it
 * look like that" unanswerable.
 *
 * **It is a module-scope function, not a factory.** Callers import `noise3`;
 * there is deliberately no `createNoise()` to call, so there is no way to
 * rebuild the table on a render — which is the shape of bug that made the
 * landing page's reference implementation tear down its particle system sixty
 * times a second.
 */

/**
 * A 512-entry permutation table: Perlin's own 256 values, doubled so the
 * lookups in `noise3` never have to mask an index twice.
 */
const PERMUTATION = [
  151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103,
  30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197,
  62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20,
  125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231,
  83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102,
  143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200,
  196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226,
  250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47,
  16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70,
  221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113,
  224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144,
  12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181,
  199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205,
  93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
];

const P = new Uint8Array(512);
for (let i = 0; i < 512; i += 1) P[i] = PERMUTATION[i & 255];

/** Perlin's quintic ease, whose second derivative is zero at both ends. */
function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

/** Dot product with one of twelve edge vectors, chosen by the low hash bits. */
function grad(hash: number, x: number, y: number, z: number): number {
  const h = hash & 15;
  const u = h < 8 ? x : y;
  const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
  return (h & 1 ? -u : u) + (h & 2 ? -v : v);
}

/** Gradient noise in `-1..1`, continuous and with a zero mean. */
export function noise3(x: number, y: number, z: number): number {
  const xi = Math.floor(x) & 255;
  const yi = Math.floor(y) & 255;
  const zi = Math.floor(z) & 255;
  const xf = x - Math.floor(x);
  const yf = y - Math.floor(y);
  const zf = z - Math.floor(z);
  const u = fade(xf);
  const v = fade(yf);
  const w = fade(zf);

  const a = P[xi] + yi;
  const aa = P[a] + zi;
  const ab = P[a + 1] + zi;
  const b = P[xi + 1] + yi;
  const ba = P[b] + zi;
  const bb = P[b + 1] + zi;

  return lerp(
    lerp(
      lerp(grad(P[aa], xf, yf, zf), grad(P[ba], xf - 1, yf, zf), u),
      lerp(grad(P[ab], xf, yf - 1, zf), grad(P[bb], xf - 1, yf - 1, zf), u),
      v,
    ),
    lerp(
      lerp(grad(P[aa + 1], xf, yf, zf - 1), grad(P[ba + 1], xf - 1, yf, zf - 1), u),
      lerp(
        grad(P[ab + 1], xf, yf - 1, zf - 1),
        grad(P[bb + 1], xf - 1, yf - 1, zf - 1),
        u,
      ),
      v,
    ),
    w,
  );
}

