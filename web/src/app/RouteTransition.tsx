/**
 * The 180ms a new screen gets.
 *
 * **Enter only, and there is no exit.** A crossfade between two screens sounds
 * right and is wrong here: holding the outgoing page for 180ms means the new
 * one cannot be painted until the old one has finished leaving, so every
 * navigation gets 180ms *slower* in exchange for looking smoother. Worse, two
 * screens overlapping need both in the layout at once, which is either a
 * `position: absolute` container that breaks the page's own scrolling or a
 * height animation, and the brief rules that out for the right reason. So the
 * old screen is replaced instantly and the new one fades and rises into place —
 * which is what a crossfade against a stable background actually looks like.
 *
 * **The sidebar is outside this on purpose.** It is chrome: it did not change,
 * and animating it would say that it did. Only the content of `<main>` is
 * wrapped — see `Shell`.
 *
 * The mechanism is a `key` on the pathname. React unmounts the old subtree and
 * mounts a new one, so the CSS animation runs from its first frame without any
 * state, timer or transition group. The cost is that a navigation remounts the
 * page component, which it already did — this is a router, and every route
 * change was already a new element.
 *
 * `?q=` and other query changes deliberately do not retrigger it. Typing in the
 * search box rewrites the query string on every submit, and a page that flashed
 * on each search would be unusable.
 */

import { useLocation } from "react-router-dom";

export function RouteTransition({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation();

  return (
    <div key={pathname} className="route-in" data-testid="route-transition">
      {children}
    </div>
  );
}
