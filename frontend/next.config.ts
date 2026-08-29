import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  /*
   * Standalone output traces the dependencies the server actually reaches and
   * emits a self-contained server.js, so the runtime image carries no
   * node_modules and no build toolchain.
   */
  output: "standalone",

  /*
   * The repository root holds the Python project and this app is a
   * subdirectory, so Next infers a workspace root it cannot be sure about and
   * warns. Stating it removes the guess and the warning.
   */
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
