import { defineConfig } from "vite";

export default defineConfig({
  // Serve the existing index.html as the root
  root: ".",
  server: {
    port: 5173,
    strictPort: true,
    // Proxy API requests to FastAPI backend (optional alternative to CORS)
    proxy: {
      "/predict": "http://localhost:8000",
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/static": "http://localhost:8000",
      "/images": "http://localhost:8000",
    },
  },
});
