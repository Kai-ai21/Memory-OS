/**
 * The two controls M9.10 added, and the two failures they exist to prevent.
 *
 * A button that is still live while its request is in flight gets pressed
 * twice, and a copy button that reports success without writing anything is
 * worse than no copy button — both are silent, and both are the kind of thing
 * that survives a visual review.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Button, CopyButton } from "./primitives";

afterEach(() => vi.unstubAllGlobals());

describe("a button that is working", () => {
  it("is disabled, keeps its label, and shows a spinner in place of the icon", async () => {
    const onClick = vi.fn();
    const { rerender } = render(
      <Button icon={<span data-testid="icon">→</span>} loading={false} onClick={onClick}>
        save the correction
      </Button>,
    );

    // At rest: the icon is there, no spinner, and it works.
    const idle = screen.getByRole("button", { name: /save the correction/i });
    expect(idle).toBeEnabled();
    expect(screen.getByTestId("icon")).toBeInTheDocument();
    expect(document.querySelector(".spinner")).toBeNull();

    rerender(
      <Button icon={<span data-testid="icon">→</span>} loading onClick={onClick}>
        save the correction
      </Button>,
    );

    const busy = screen.getByRole("button", { name: /save the correction/i });

    // Disabled, and announced as busy rather than only looking it.
    expect(busy).toBeDisabled();
    expect(busy).toHaveAttribute("aria-busy", "true");

    // The spinner has taken the icon's place — both halves matter. A spinner
    // added *beside* the icon changes the width of the button mid-click.
    expect(document.querySelector(".spinner")).toBeInTheDocument();
    expect(screen.queryByTestId("icon")).not.toBeInTheDocument();

    // **The label is unchanged.** This is the half a "correcting…" swap gets
    // wrong: the reader loses the name of the thing they just asked for at
    // exactly the moment they are checking whether it happened.
    expect(busy).toHaveTextContent("save the correction");

    // And the press it was disabled to prevent does not land.
    await userEvent.click(busy);
    expect(onClick).not.toHaveBeenCalled();
  });
});

describe("copying a value", () => {
  it("writes the full value to the clipboard and confirms on the icon", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });

    const hash = "9f2c1ab4e7d3f8091b6a5c4d2e1f0a9b8c7d6e5f";
    render(<CopyButton value={hash} label="content hash" />);

    await userEvent.click(screen.getByRole("button", { name: "Copy content hash" }));

    // The whole hash, not the twelve characters the page displays. A copy
    // button that hands back the truncation produces a value that looks right
    // and matches nothing.
    expect(writeText).toHaveBeenCalledExactlyOnceWith(hash);

    // Confirmed on the control that was pressed, by its accessible name
    // changing — no toast, and nothing to dismiss.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Copied content hash" })).toBeInTheDocument(),
    );
  });

  it("claims nothing when the clipboard refuses", async () => {
    // Permission-gated and unavailable on insecure origins, so this is a real
    // path rather than a defensive one — and a tick for a copy that did not
    // happen is the worst outcome available here.
    const writeText = vi.fn().mockRejectedValue(new Error("denied"));
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });

    render(<CopyButton value="abc" label="path" />);
    await userEvent.click(screen.getByRole("button", { name: "Copy path" }));

    expect(screen.getByRole("button", { name: "Copy path" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copied path" })).not.toBeInTheDocument();
  });
});
