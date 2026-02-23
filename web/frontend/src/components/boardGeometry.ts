export const CELL = 80
export const PAD = 50
export const SVG = 4 * CELL + 2 * PAD  // 420

/** Convert square index (0-24) to SVG (x, y). Row 0 = bottom of screen. */
export function sqXY(sq: number): [number, number] {
  const r = Math.floor(sq / 5)
  const c = sq % 5
  return [PAD + c * CELL, PAD + (4 - r) * CELL]
}
