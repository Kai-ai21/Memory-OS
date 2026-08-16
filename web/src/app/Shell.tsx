/**
 * The frame: a persistent sidebar, a reading column, and the keyboard model.
 *
 * **The shortcuts are registered here rather than per page.** `/` used to be
 * bound inside the search view, which meant it worked on exactly one of
 * fourteen routes and silently did nothing on the other thirteen — the failure
 * mode of a keyboard model that lives in a component. Bound at the shell, `/`
 * means the same thing everywhere: go and search. Where there is no box on
 * screen, it navigates to the one there is.
 *
 * Below 768px the sidebar becomes a drawer over the content rather than a
 * squeezed column. A 15rem nav beside a 320px article leaves neither of them
 * usable, and the nav is the half you need less often.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

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
      {/* Skip link: the sidebar is a dozen tab stops, and every page repeats
          them. First focusable thing in the document, visible once focused. */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:border focus:border-rule-strong focus:bg-raised focus:px-3 focus:py-1.5 focus:text-xs"
        onClick={(event) => {
          event.preventDefault();
          main.current?.focus();
        }}
      >
        skip to content
      </a>

      {/* --- The bar that only exists on small screens ------------------- */}
      <header className="flex items-center justify-between border-b border-rule-strong px-4 py-2 md:hidden">
        <button
          type="button"
          className="btn"
          aria-expanded={drawerOpen}
          aria-controls="sidebar"
          onClick={() => setDrawerOpen((current) => !current)}
        >
          menu
        </button>
        <span className="display text-base">Memory OS</span>
        <button type="button" className="btn" onClick={() => setPaletteOpen(true)}>
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
        className={`fixed inset-y-0 left-0 z-40 w-60 border-r border-rule-strong transition-transform md:sticky md:top-0 md:z-auto md:h-dvh md:w-(--width-sidebar) md:translate-x-0 ${
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
        className="min-w-0 flex-1 px-5 py-6 outline-none sm:px-8 md:px-12 md:py-12"
      >
        <div className="mx-auto max-w-5xl">{children}</div>
      </main>

      <CommandPalette open={paletteOpen} onClose={closePalette} />
    </div>
  );
}
