/**
 * The frame: the lit void, a persistent sidebar, and the keyboard model.
 *
 * `BackgroundLayer` is mounted here and is the first thing in the tree. It
 * paints the two pale radials that the one glass element — the NEW
 * CONVERSATION button — frosts. Nothing else in the application is
 * translucent, so that component is the only reason it exists; it is kept
 * isolated so a canvas or a cursor-following field can replace its contents
 * later without touching this file.
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
import { BackgroundLayer } from "./BackgroundLayer";
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
      <BackgroundLayer />

      {/* Skip link: the sidebar is a dozen tab stops, and every page repeats
          them. First focusable thing in the document, visible once focused. */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:border focus:border-rule-strong focus:bg-surface focus:px-3 focus:py-1.5 focus:text-xs"
        onClick={(event) => {
          event.preventDefault();
          main.current?.focus();
        }}
      >
        skip to content
      </a>

      {/* --- The bar that only exists on small screens ------------------- */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-rule bg-surface px-4 py-2 md:hidden">
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
        <span className="display text-base font-bold tracking-[0.14em]">
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
