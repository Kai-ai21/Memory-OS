/**
 * The frame: the lit void, a persistent sidebar, and the keyboard model.
 *
 * **The glows are structural, not decorative.** Two large radial gradients are
 * painted here, fixed to the viewport, behind everything. Every panel in this
 * application is white at 3% with a 12px backdrop blur, and a translucent panel
 * over a flat colour is just a slightly lighter flat colour — the blur has
 * nothing to blur and the whole design collapses to grey boxes. These are what
 * it is translucent *of*: the same panel reads cyan at the top left and magenta
 * at the bottom right, which is what makes the surfaces look like glass rather
 * than like fills. Delete them and the design system stops working.
 *
 * They drift, slowly, on a 20s alternating float. Slow enough that it is not an
 * animation you notice — it reads as light moving rather than as an element
 * moving — and it stops entirely under `prefers-reduced-motion`.
 *
 * **The shortcuts are registered here rather than per page.** `/` used to be
 * bound inside the search view, which meant it worked on exactly one of fifteen
 * routes and silently did nothing on the other fourteen — the failure mode of a
 * keyboard model that lives in a component. Bound at the shell, `/` means the
 * same thing everywhere: go and search. Where there is no box on screen, it
 * navigates to the one there is.
 *
 * Below 768px the sidebar becomes a drawer over the content rather than a
 * squeezed column. A 17.5rem nav beside a 320px article leaves neither of them
 * usable, and the nav is the half you need less often.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { Icon } from "../components/Icon";
import { CommandPalette } from "./CommandPalette";
import { Sidebar } from "./Sidebar";

export function Shell({ children }: { children: React.ReactNode }) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const main = useRef<HTMLElement>(null);

  const closePalette = useCallback(() => setPaletteOpen(false), []);

  // Navigating from the drawer closes it. Without this the nav stays over the
  // page you just asked for, which reads as a click that did nothing.
  useEffect(() => setDrawerOpen(false), [location.pathname]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable === true;

      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        // Claimed even while typing: ⌘K is not a character, and a search box
        // that swallowed it would be the one place the shortcut fails.
        event.preventDefault();
        setPaletteOpen((current) => !current);
        return;
      }

      // `/` is a character, so it is only a shortcut when nothing is being
      // typed into — otherwise a path query could never be entered.
      if (event.key === "/" && !typing) {
        event.preventDefault();
        const box = document.querySelector<HTMLInputElement>("[data-search-input]");
        if (box) box.focus();
        else navigate("/search");
        return;
      }

      if (event.key === "Escape") {
        // Both dismissals live here rather than in the components, so that
        // "Esc closes whatever is over the page" is one rule with one
        // implementation. `<dialog>` also closes itself on Esc natively and
        // routes that through `onCancel`; the two agree, and setting the same
        // state twice is harmless.
        if (drawerOpen) setDrawerOpen(false);
        setPaletteOpen(false);
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, drawerOpen]);

  return (
    <div className="min-h-dvh md:flex">
      <Glows />

      {/* Skip link: the sidebar is a dozen tab stops, and every page repeats
          them. First focusable thing in the document, visible once focused. */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:border focus:border-rule-strong focus:bg-float focus:px-3 focus:py-1.5 focus:text-xs"
        onClick={(event) => {
          event.preventDefault();
          main.current?.focus();
        }}
      >
        skip to content
      </a>

      {/* --- The bar that only exists on small screens ------------------- */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-rule bg-nav px-4 py-2 backdrop-blur-xl md:hidden">
        <button
          type="button"
          className="btn flex items-center gap-2"
          aria-expanded={drawerOpen}
          aria-controls="sidebar"
          onClick={() => setDrawerOpen((current) => !current)}
        >
          <Icon name="menu" size={16} />
          menu
        </button>
        <span className="display glow-cyan text-base font-bold tracking-[0.14em]">
          MEMO
        </span>
        <button
          type="button"
          className="btn flex items-center gap-2"
          onClick={() => setPaletteOpen(true)}
        >
          <Icon name="search" size={16} />
          search
        </button>
      </header>

      {/* --- The sidebar ------------------------------------------------- */}
      {drawerOpen ? (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 z-30 bg-scrim md:hidden"
          onClick={() => setDrawerOpen(false)}
        />
      ) : null}
      <aside
        id="sidebar"
        className={`fixed inset-y-0 left-0 z-40 w-70 border-r border-rule transition-transform md:sticky md:top-0 md:z-auto md:h-dvh md:w-(--width-sidebar) md:translate-x-0 ${
          drawerOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Sidebar onNavigate={() => setDrawerOpen(false)} />
      </aside>

      {/* --- The content ------------------------------------------------- */}
      <main
        id="main"
        ref={main}
        tabIndex={-1}
        className="min-w-0 flex-1 px-5 py-6 outline-none sm:px-8 md:px-8 md:py-10"
      >
        <div className="mx-auto max-w-300">{children}</div>
      </main>

      <CommandPalette open={paletteOpen} onClose={closePalette} />
    </div>
  );
}

/**
 * The light behind everything.
 *
 * Fixed rather than absolute, so the light stays where it is while the page
 * scrolls under it — a glow that scrolls with the content reads as a coloured
 * shape *in* the document, which is exactly what it must not look like.
 *
 * `pointer-events-none` and `aria-hidden` throughout: this is two divs of pure
 * light, and neither the mouse nor a screen reader should ever find them.
 * `overflow-hidden` on the wrapper is what stops the two 50vw circles, which
 * are deliberately positioned off-screen at their corners, from giving the
 * document a horizontal scrollbar.
 */
function Glows() {
  return (
    <div
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
      aria-hidden
      data-testid="glows"
    >
      <div
        className="absolute -top-[10%] -left-[10%] size-[55vw] animate-[drift_20s_ease-in-out_infinite_alternate] rounded-full opacity-70 blur-[80px]"
        style={{ backgroundImage: "var(--glow-magenta)" }}
      />
      <div
        className="absolute -right-[10%] -bottom-[15%] size-[45vw] animate-[drift_20s_ease-in-out_-10s_infinite_alternate] rounded-full opacity-70 blur-[80px]"
        style={{ backgroundImage: "var(--glow-cyan)" }}
      />
    </div>
  );
}
