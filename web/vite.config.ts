import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
    // The generated schema is 1,000 lines of types and nothing else; counting it
    // as uncovered source would make the number meaningless.
    coverage: { exclude: ["src/api/schema.d.ts", "src/api/openapi.json"] },
  },
});
