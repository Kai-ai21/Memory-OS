/**
 * The icon set, drawn rather than downloaded.
 *
 * The reference links Material Symbols from `fonts.googleapis.com`. That is a
 * network request per page load in a tool whose entire premise is that it works
 * with no network, and the alternative — vendoring the font — is four megabytes
 * of variable font for the fourteen glyphs this interface actually uses. So
 * these are hand-drawn on Material's own 24px grid and shipped as part of the
 * JavaScript, which costs about two kilobytes.
 *
 * Stroke rather than fill, at 1.5px on a 24px grid: a filled glyph at this size
 * reads as a solid blob, and the stroke weight is what keeps an icon and its
 * label looking like they belong to each other.
 *
 * **M9.9 moved the sidebar off this set and it is not a repudiation of it.**
 * The nav is thirteen glyphs that have to be told apart at a glance beside
 * 14.5px prose, and lucide draws that set consistently because one person drew
 * all of it — see `app/routes.ts`, which now holds the components themselves.
 * What is left here is what has always been true of these: they are the glyphs
 * this application draws *inside* its own screens, next to mono, at 14 to 17px,
 * where the set is small enough to stay coherent by hand and a dependency would
 * be a dependency for five drawings. The eight this file gained in M9.8 for the
 * sidebar left with it.
 *
 * `aria-hidden` throughout and never the only label. Each of these appears
 * beside real text — a nav item, a button — so announcing the icon as well
 * would make a screen reader say everything twice.
 */

export type IconName =
  | "search"
  | "timeline"
  | "graph"
  | "decisions"
  | "insights"
  | "sources"
  | "chat"
  | "settings"
  | "help"
  | "add"
  | "attach"
  | "send"
  | "document"
  | "code"
  | "tune"
  | "close"
  | "menu";

/** The path data, on a 24×24 grid with a 2px optical margin. */
const PATHS: Record<IconName, React.ReactNode> = {
  search: (
    <>
      <circle cx="11" cy="11" r="6" />
      <path d="M15.5 15.5 20 20" />
    </>
  ),
  timeline: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 7.5V12l3 2" />
    </>
  ),
  graph: (
    <>
      <circle cx="12" cy="12" r="2.5" />
      <circle cx="5" cy="7" r="2" />
      <circle cx="19" cy="7" r="2" />
      <circle cx="6" cy="18" r="2" />
      <circle cx="18" cy="18" r="2" />
      <path d="m10.2 10.4-3.4-2.2m6.9 2.2 3.5-2.2m-6.9 5 3.5 2.2m-3.4-2.2-3.6 2.2" />
    </>
  ),
  decisions: (
    <>
      <rect x="4" y="5" width="16" height="14" rx="2" />
      <path d="m8 12 2.2 2.2L15 9.5" />
    </>
  ),
  insights: (
    <>
      <path d="M9.5 17.5h5M10 20.5h4" />
      <path d="M12 3.5a5.8 5.8 0 0 0-3.4 10.5c.5.4.9 1 .9 1.6h5c0-.6.4-1.2.9-1.6A5.8 5.8 0 0 0 12 3.5Z" />
    </>
  ),
  sources: (
    <>
      <ellipse cx="12" cy="6.5" rx="7" ry="2.8" />
      <path d="M5 6.5v11c0 1.6 3.1 2.8 7 2.8s7-1.2 7-2.8v-11" />
      <path d="M5 12c0 1.6 3.1 2.8 7 2.8s7-1.2 7-2.8" />
    </>
  ),
  chat: (
    <path d="M20 12.5c0 3.6-3.6 6.5-8 6.5a9.7 9.7 0 0 1-2.6-.35L5 20.5l1.2-3.2A6.1 6.1 0 0 1 4 12.5C4 8.9 7.6 6 12 6s8 2.9 8 6.5Z" />
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="2.8" />
      <path d="M12 3.5h0l.5 2.3 2.2.9 2-1.2 1.8 1.8-1.2 2 .9 2.2 2.3.5v2.5l-2.3.5-.9 2.2 1.2 2-1.8 1.8-2-1.2-2.2.9-.5 2.3H11l-.5-2.3-2.2-.9-2 1.2-1.8-1.8 1.2-2-.9-2.2-2.3-.5v-2.5l2.3-.5.9-2.2-1.2-2 1.8-1.8 2 1.2 2.2-.9L11 3.5Z" />
    </>
  ),
  help: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M9.8 9.6a2.3 2.3 0 1 1 2.9 2.6c-.5.2-.7.7-.7 1.2v.6" />
      <path d="M12 16.6h.01" />
    </>
  ),
  add: <path d="M12 5.5v13M5.5 12h13" />,
  attach: (
    <path d="M17 10.5 11.2 16.3a3 3 0 0 1-4.3-4.3l6.6-6.6a2 2 0 0 1 2.9 2.9l-6.6 6.6a1 1 0 0 1-1.4-1.4l5.8-5.8" />
  ),
  send: <path d="M12 19V6m0 0-5 5m5-5 5 5" />,
  document: (
    <>
      <path d="M14 3.5H7a1.5 1.5 0 0 0-1.5 1.5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8Z" />
      <path d="M14 3.5V8h4.5" />
    </>
  ),
  code: <path d="m9 8.5-4 3.5 4 3.5m6-7 4 3.5-4 3.5" />,
  tune: (
    <>
      <path d="M4 8h9m3 0h4M4 16h4m3 0h9" />
      <circle cx="14.5" cy="8" r="1.8" />
      <circle cx="9.5" cy="16" r="1.8" />
    </>
  ),
  close: <path d="m6.5 6.5 11 11m0-11-11 11" />,
  menu: <path d="M4.5 7.5h15m-15 4.5h15m-15 4.5h15" />,
};

interface Props {
  name: IconName;
  /** Pixel size. 17 for a button, 14 to 16 inline beside mono text. */
  size?: number;
  className?: string;
}

export function Icon({ name, size = 20, className }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
      focusable="false"
    >
      {PATHS[name]}
    </svg>
  );
}
