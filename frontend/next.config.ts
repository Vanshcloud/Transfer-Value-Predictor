import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
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
