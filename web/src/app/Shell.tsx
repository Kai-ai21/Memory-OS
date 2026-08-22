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
 * **`?` opens the shortcuts sheet, and that sheet is the reason the rest of
 * these are worth having.** Four of the five bindings below were, until M9.10,
 * documented nowhere a user could see — see `ShortcutsSheet.tsx`.
 *
 * **The shortcuts are registered here rather than per page.** `/` used to be
 * bound inside the search view, which meant it worked on exactly one of fifteen
 * routes and silently did nothing on the other fourteen — the failure mode of a
 * keyboard model that lives in a component. Bound at the shell, `/` means the
 * same thing everywhere: go and search. Where there is no box on screen, it
 * navigates to the one there is.
 *
 * Below 768px the sidebar becomes a drawer over the content rather than a
 * squeezed column. A 16.5rem nav beside a 320px article leaves neither of them
 * usable, and the nav is the half you need less often.
 *
 * **M9.8 unsticks the sidebar from the edge.** It is a panel floating in a 12px
 * margin now rather than a column flush against the viewport, so the `aside`
 * here owns the inset and the radius-clipping and `Sidebar` owns the glass. The
 * shell also owns whether it is shown at all: the panel's collapse control is a
 * handle into this component's state, because a component cannot un-render
 * itself and the button that brings it back has to live somewhere the panel is
 * not.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { PanelLeft } from "lucide-react";

import { Icon } from "../components/Icon";
import { BackgroundLayer } from "./BackgroundLayer";
import { CommandPalette } from "./CommandPalette";
import { ShortcutsSheet } from "./ShortcutsSheet";
import { SplitPanel } from "./SplitPanel";
import { SplitProvider } from "./SplitProvider";
import { useCloseSplitOnDuplicate, useSplit, useSplitShortcuts } from "../lib/split";
import { Sidebar } from "./Sidebar";

/**
 * The provider has to sit outside the frame that reads it.
 *
 * `Shell` itself calls `useSplit` (through `useSplitShortcuts` and the panel),
 * so it cannot also be the component that provides the context — a component
 * never sees its own provider. Hence the thin wrapper.
 */
export function Shell({ children }: { children: React.ReactNode }) {
  return (
    <SplitProvider>
      <ShellFrame>{children}</ShellFrame>
    </SplitProvider>
  );
}

function ShellFrame({ children }: { children: React.ReactNode }) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  /* Session state rather than a stored preference, deliberately. Hiding the
     nav is something you do to get a wide view of one page for a minute, not a
     way you want the application to start — and a nav that is missing on load
     because of something you did last Tuesday is a bug report. */
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const main = useRef<HTMLElement>(null);

  const closePalette = useCallback(() => setPaletteOpen(false), []);

  // ⌘\ and the split's own Esc. Assembled here with the rest of the keyboard
  // model even though the binding lives beside the state it needs.
  useSplitShortcuts();
  useCloseSplitOnDuplicate(location.pathname);
  const splitOpen = useSplit().memoryId !== null;
  const closeShortcuts = useCallback(() => setShortcutsOpen(false), []);

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

      /* `?` opens the sheet, and like `/` it is a character — so it is only a
         shortcut when nothing is being typed into. Guarded on the modifiers
         too: `⌘?` and `Ctrl+?` are browser and OS bindings, and claiming them
         would be this application taking a key it does not own.

         Matched on `event.key` rather than on shift-plus-slash, because `?` is
         an unshifted key on several layouts and a `shiftKey` check would make
         the shortcut unreachable on all of them. */
      if (event.key === "?" && !typing && !event.metaKey && !event.ctrlKey) {
        event.preventDefault();
        setShortcutsOpen((current) => !current);
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
        setShortcutsOpen(false);
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
          onClick={() => {
            // The drawer and the desktop collapse are one panel in two
            // positions, so asking for the menu on a phone has to undo a
            // collapse made on a wider window.
            setCollapsed(false);
            setDrawerOpen((current) => !current);
          }}
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
      {/* 12px off the top, left and bottom at every width — the drawer is the
          same floating panel arriving from off-screen rather than a different
          object. `-translate-x-[110%]` rather than `-full` so that the shadow
          and the 12px margin go with it; at `-full` the soft edge of the shadow
          stays visible against the page. */}
      {collapsed ? null : (
        <aside
          id="sidebar"
          className={`fixed inset-y-3 left-3 z-40 w-(--width-sidebar) transition-transform duration-(--dur-travel) ease-(--ease-out) md:sticky md:top-3 md:z-auto md:my-3 md:ml-3 md:h-[calc(100dvh-1.5rem)] md:translate-x-0 ${
            drawerOpen ? "translate-x-0" : "-translate-x-[110%]"
          }`}
        >
          <Sidebar
            onNavigate={() => setDrawerOpen(false)}
            onShortcuts={() => setShortcutsOpen(true)}
            onCollapse={() => {
              setDrawerOpen(false);
              setCollapsed(true);
            }}
          />
        </aside>
      )}

      {/* What brings it back. One 32px control in the corner the panel left,
          and it only exists while the panel does not. */}
      {collapsed ? (
        <button
          type="button"
          className="panel icon-button fixed top-3 left-3 z-40 hidden md:inline-flex"
          aria-label="Show navigation"
          title="Show navigation"
          onClick={() => setCollapsed(false)}
        >
          <PanelLeft size={18} strokeWidth={1.5} />
        </button>
      ) : null}

      {/* --- The content ------------------------------------------------- */}
      <main
        id="main"
        ref={main}
        tabIndex={-1}
        /* `overflow-y-auto` and a fixed height only once the split is open.
           The page scrolls the window normally — that is the right behaviour
           for a document and it is what every route expects — but two panes
           that scroll the window together are not a split, they are one long
           column cut in half. */
        className={`min-w-0 flex-1 px-5 py-6 outline-none sm:px-8 md:px-8 md:py-10 ${
          splitOpen ? "h-dvh overflow-y-auto" : ""
        }`}
      >
        {/* Marked as a shelter for the background field: the cursor trail is
            multiplied to zero inside this box and climbs back to full strength
            in the gutters and behind the sidebar. Measured off the DOM by
            `lib/mask`, so this stays true when the column changes width and
            nothing has to be told about it. */}
        <div className="mx-auto max-w-300" data-particle-shelter>
          {children}
        </div>
      </main>

      {/* Below `md` the panel would leave neither pane usable — the sidebar
          already becomes a drawer at that width for the same reason. */}
      <div className="hidden md:contents">
        <SplitPanel />
      </div>

      <CommandPalette open={paletteOpen} onClose={closePalette} />
      <ShortcutsSheet open={shortcutsOpen} onClose={closeShortcuts} />
    </div>
  );
}
