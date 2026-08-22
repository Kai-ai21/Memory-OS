import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { App } from "./App";
import { applyDensity, readDensity } from "./lib/density";
import "./index.css";

/* Before the first render rather than in an effect. Applied from an effect the
   page paints once at comfortable and then reflows, which is the flash of
   wrong layout every stored theme setting has to avoid. */
applyDensity(readDensity());

/**
 * TanStack Query is the only state management here, and that is the whole point:
 * every piece of state in this app is server state. Search results, memory
 * details, stats — all of it is a cached view of something the API owns. Client
 * state is a search box and some filters, and those live in the URL.
 *
 * Retries are off. Against a local API a failure means the API is down or the
 * request was wrong, and three silent retries turn a two-second "start the API"
 * into a ten-second wait for the same answer.
 */
const client = new QueryClient({
  defaultOptions: {
    queries: { retry: false, refetchOnWindowFocus: false },
    mutations: { retry: false },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
