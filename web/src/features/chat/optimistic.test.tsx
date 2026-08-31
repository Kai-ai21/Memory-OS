/**
 * The optimistic message, and the state that matters: the one that failed.
 *
 * Showing your own words immediately is easy and every optimistic UI does it.
 * The part that gets skipped is what happens when the send does not land — and
 * a message drawn instantly and then silently removed is *worse* than one that
 * never appeared, because the reader watched it show up and believes it was
 * sent.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChatPage } from "./ChatPage";
import { SHELL_ROUTES, renderWithProviders, stubFetch } from "../../test/harness";

afterEach(() => vi.unstubAllGlobals());

/** Everything the chat asks for, with `POST /chat` failing. */
function stubFailingSend() {
  return stubFetch([
    ...SHELL_ROUTES,
    { match: "/chat/messages", body: [] },
    { match: "/chat", body: { detail: "upstream is down" }, status: 500 },
  ]);
}

describe("a message that did not send", () => {
  it("renders in the warn state, keeps the text, and offers a retry", async () => {
    const calls = stubFailingSend();
    renderWithProviders(<ChatPage />, { route: "/" });

    const box = await screen.findByLabelText("Message");
    await userEvent.type(box, "postgres lease semantics");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    const failed = await screen.findByTestId("pending-failed");

    // `warn`, not `deny`: nothing is broken and nothing was refused — it did
    // not send, and it can be sent again.
    expect(failed.className).toContain("border-warn");
    expect(screen.getByText("not sent")).toBeInTheDocument();

    // **The words survive.** Losing what somebody typed because the network
    // dropped is the actual damage; the failure itself is recoverable.
    expect(failed).toHaveTextContent("postgres lease semantics");

    // Announced, because it appeared away from where the eye was.
    expect(failed).toHaveAttribute("role", "alert");

    const sends = () => calls.filter((call) => call.method === "POST").length;
    const before = sends();

    await userEvent.click(screen.getByTestId("pending-retry"));

    // Retry re-sends the same text rather than clearing the box and hoping.
    await waitFor(() => expect(sends()).toBe(before + 1));
    expect(calls.at(-1)?.body).toMatchObject({ text: "postgres lease semantics" });
  });

  it("can be discarded, and takes the failure away with it", async () => {
    stubFailingSend();
    renderWithProviders(<ChatPage />, { route: "/" });

    const box = await screen.findByLabelText("Message");
    await userEvent.type(box, "a thought");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    await screen.findByTestId("pending-failed");
    await userEvent.click(screen.getByTestId("pending-discard"));

    await waitFor(() =>
      expect(screen.queryByTestId("pending-failed")).not.toBeInTheDocument(),
    );
  });
});
