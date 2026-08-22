/**
 * ⌘Enter submits the form you are in.
 *
 * **A textarea has no submit key of its own**, which is the whole problem. A
 * single-line `<input>` submits its form on Enter and always has; a `<textarea>`
 * treats Enter as a newline, correctly, and so a form built around one has no
 * keyboard route to its own submit button. The convention every application
 * with a compose box has landed on is ⌘Enter, and it is worth following for the
 * usual reason — somebody arriving here will already have the habit.
 *
 * Attached to the `<form>` rather than to each field, so it covers the textarea,
 * every input beside it, and anything added later. Keyboard events bubble, so
 * one handler at the top is the same behaviour as a handler on each child minus
 * the chance of forgetting one.
 *
 * `requestSubmit` rather than `submit`: `submit()` bypasses the `submit` event
 * entirely, which means it skips both native validation and the `onSubmit`
 * handler that every form in this application does its actual work in. They are
 * one character apart and the wrong one silently does nothing here.
 *
 * The chat composer keeps bare Enter as well — see `ChatPage`. The two do not
 * conflict: Enter sends, shift-Enter is a newline, and ⌘Enter also sends, so
 * the habit from other applications works without taking anything away.
 */
export function submitOnCmdEnter(event: React.KeyboardEvent<HTMLFormElement>): void {
  if (event.key !== "Enter") return;
  if (!event.metaKey && !event.ctrlKey) return;

  const form = event.currentTarget;
  // A form with nothing to submit to — every form here has a submit button, but
  // `requestSubmit` throws rather than no-ops if that ever stops being true.
  if (typeof form.requestSubmit !== "function") return;

  event.preventDefault();
  form.requestSubmit();
}
