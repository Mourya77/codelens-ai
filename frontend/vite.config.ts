import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// All API calls use relative `/api/...` paths. In production the
// FastAPI backend serves this build from the same origin, so no
// proxying is needed; in dev, Vite proxies /api to the backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
