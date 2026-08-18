/**
 * `cn` — join class names, dropping the falsy ones.
 *
 * **Deliberately not `clsx` + `tailwind-merge`.** That pair earns its keep in a
 * component library where a caller's `p-4` has to beat a default `p-2`, which
 * requires actually parsing Tailwind's grammar. Nothing here does that: the one
 * component taking a `className` merges a fixed base with a caller's optional
 * extra, and for that the whole job is "skip the undefined ones". Two
 * dependencies and a bundle entry for a conditional join would be the cost of
 * the idiom rather than of the behaviour.
 *
 * If a real precedence conflict ever appears, this is the file to replace.
 */
export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}
