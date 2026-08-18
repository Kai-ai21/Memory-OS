/**
 * The front door: a wordmark, a line of copy, a sign-in form that does not
 * sign anybody in, and the one link that actually goes somewhere.
 *
 * **The form is scaffolding and says so twice.** There is no user system in
 * this project — no users table, no sessions, nothing that could accept a
 * password — so a form that looked operable would be a lie told by an
 * interface, and the worst version of that lie is the one that takes a
 * password before admitting it. It is wrapped in a disabled `<fieldset>`,
 * which is the one construct that makes "inert" a fact about the document
 * rather than a styling choice: every control inside is unfocusable and
 * unsubmittable, and assistive technology is told so without a single
 * `aria-*` attribute.
 *
 * **Which makes the accent rule do the work here.** Rule 1 of this theme is
 * that the accent means "you can do something here", and on this page exactly
 * one thing qualifies. CONTINUE is the loud element and is dead; `Open MEMO`
 * is quiet and is live. Left alone that is precisely backwards, so the button
 * carries the disabled treatment — dimmed, default cursor — and the link
 * carries the accent, an underline and the arrow. The hierarchy the eye reads
 * and the hierarchy the mouse discovers end up agreeing.
 *
 * The card is the only element on the page with a shadow. Everything else sits
 * flat on the canvas, which is what keeps the one raised thing meaning
 * "raised" rather than being one of several.
 */

import { Link } from "react-router-dom";

import { FluidParticles } from "./FluidParticles";

export function WelcomePage() {
  return (
    <FluidParticles>
      <main
        /* The shelter, with its own feather: the field fades to nothing over
           this box and takes 280px to reach full strength again, which on this
           page means the whole centre is quiet and the edges are not. Wider
           than the app's 160 because there is far more room here to spend. */
        data-particle-shelter="280"
        className="flex w-full max-w-110 flex-col items-center gap-8 px-6 py-12 text-center"
      >
        {/* --- The mark ---------------------------------------------------- */}
        <header className="flex flex-col items-center gap-2">
          <h1 className="display text-headline-xl font-bold tracking-[0.06em]">MEMO</h1>
          <p className="meta-label text-ink-3">Kailaas OS</p>
        </header>

        <p className="font-prose text-body-sm text-ink-2">
          Everything you&rsquo;ve thought, and what it connects to.
        </p>

        {/* --- The shell of a sign-in -------------------------------------- */}
        {/* The card is narrower than the copy above it. A sign-in form as wide
            as a sentence reads as a page rather than as an object on one. */}
        <div className="flex w-full max-w-90 flex-col gap-3">
        <form
          className="panel panel-raised w-full p-5 text-left"
          /* Belt and braces. A disabled fieldset cannot submit, so this only
             fires if somebody removes the fieldset and forgets why it was
             there — at which point a page reload is the least useful outcome. */
          onSubmit={(event) => event.preventDefault()}
        >
          <fieldset disabled className="flex flex-col gap-3">
            <legend className="sr-only">Sign in to MEMO (not yet active)</legend>

            <label className="flex flex-col gap-1.5">
              <span className="meta-label">email</span>
              <input type="email" name="email" autoComplete="off" className="field" />
            </label>

            <label className="flex flex-col gap-1.5">
              <span className="meta-label">password</span>
              <input
                type="password"
                name="password"
                autoComplete="off"
                className="field"
              />
            </label>

            <button type="submit" className="btn-primary mt-1 w-full">
              Continue
            </button>
          </fieldset>
        </form>

        <p className="meta text-ink-3">
          Sign-in isn&rsquo;t active yet &mdash; MEMO runs locally on your machine.
        </p>
        </div>

        {/* --- The path that works ----------------------------------------- */}
        <Link
          to="/"
          /* The only saturated accent on the page. CONTINUE above it wears the
             same hue at the disabled treatment's 45%, so "full strength means
             live" is legible without reading a word — rule 1, doing the job it
             was reserved for. Set a step larger than the copy so the eye finds
             it after the card rather than never. */
          className="font-prose text-accent text-body-md font-medium underline decoration-1 underline-offset-4 transition-[text-decoration-thickness] hover:decoration-2"
        >
          Open MEMO &rarr;
        </Link>
      </main>
    </FluidParticles>
  );
}
