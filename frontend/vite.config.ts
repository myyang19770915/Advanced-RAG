import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendBase = process.env.VITE_API_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: backendBase,
        changeOrigin: true,
        configure: (proxy) => {
          // For SSE / streaming responses:
          // 1. Forward the no-buffer hint headers.
          // 2. Disable Nagle's algorithm on the client-facing socket so each
          //    chunk is sent over the wire immediately without TCP coalescing.
          proxy.on("proxyRes", (proxyRes, req, res) => {
            const isSSE =
              (req.headers.accept ?? "").includes("text/event-stream") ||
              (proxyRes.headers["content-type"] ?? "").includes("text/event-stream");
            if (isSSE) {
              proxyRes.headers["x-accel-buffering"] = "no";
              proxyRes.headers["cache-control"] = "no-cache";
              proxyRes.headers["connection"] = "keep-alive";
              // Flush each chunk to the remote browser immediately.
              const socket = (res as any).socket;
              if (socket && typeof socket.setNoDelay === "function") {
                socket.setNoDelay(true);
              }
            }
          });
        },
      },
    },
  },
});
