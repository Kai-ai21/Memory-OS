/**
 * The front door: a wordmark, a line of copy, a sign-in form that does not
 * sign anybody in, and the one link that actually goes somewhere.
 *
 * **M11.0 made the form real.** Until this milestone there was no user system
 * to sign in to, the fieldset was disabled, and the page said so underneath —
 * because a form that looked operable would have been a lie told by an
 * interface, and the worst version of that lie is the one that takes a
 * password before admitting it. All of that is gone: it posts to `/auth/login`
 * and a success lands you on `/`.
 *
 * **One error message for every failure.** An unknown address and a wrong
 * password produce the same sentence, because the server produces the same
 * response — anything else lets somebody read which addresses have accounts
 * off the error text. The only failure worded differently is the rate limit,
 * which is not about the credentials at all.
 *
 * Nothing is stored here. The session is an `HttpOnly` cookie the browser
 * holds and JavaScript cannot read, which is why there is no token in
 * `localStorage`, no context provider holding one, and nothing to clear on
 * logout but the server's own record.
 *
 * The card is the only element on the page with a shadow. Everything else sits
 * flat on the canvas, which is what keeps the one raised thing meaning
 * "raised" rather than being one of several.
 */

import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { ApiError, NetworkError, api } from "../../api/client";
import { FluidParticles } from "./FluidParticles";

export function WelcomePage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await api.login(email, password);
      // `replace`, so the back button does not return to a sign-in page the
      // session has already passed.
      navigate("/", { replace: true });
    } catch (cause) {
      if (cause instanceof NetworkError) {
        setError("Could not reach MEMO. Is the API running?");
      } else if (cause instanceof ApiError && cause.status === 429) {
        // The one failure that is not about the credentials, and saying so
        // stops somebody retyping a password that was never the problem.
        setError("Too many attempts. Try again in a few minutes.");
      } else {
        setError("Incorrect email or password");
      }
    } finally {
      setBusy(false);
    }
  }

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
        <form className="panel panel-raised w-full p-5 text-left" onSubmit={submit}>
          <fieldset disabled={busy} className="flex flex-col gap-3">
            <legend className="sr-only">Sign in to MEMO</legend>

            <label className="flex flex-col gap-1.5">
              <span className="meta-label">email</span>
              <input
                type="email"
                name="email"
                autoComplete="username"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className="field"
              />
            </label>

            <label className="flex flex-col gap-1.5">
              <span className="meta-label">password</span>
              <input
                type="password"
                name="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="field"
              />
            </label>

            <button type="submit" className="btn-primary mt-1 w-full">
              {busy ? "Signing in…" : "Continue"}
            </button>
          </fieldset>
        </form>

        {/* `role="alert"` so the message is announced rather than only drawn:
            somebody using a screen reader gets no other signal that the form
            came back. `deny` rather than the accent — this is not an action. */}
        {error ? (
          <p role="alert" className="meta text-deny" data-testid="signin-error">
            {error}
          </p>
        ) : null}
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
