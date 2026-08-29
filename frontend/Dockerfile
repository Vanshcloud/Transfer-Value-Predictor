# ============================================================
# Dashboard image
# ============================================================
# Three stages so the runtime image carries neither node_modules nor the
# toolchain — Next's standalone output bundles exactly the dependencies the
# server actually reaches.

FROM node:24-slim AS deps
WORKDIR /app
# `npm ci` rather than `npm install`: it installs the lockfile exactly and
# fails if package.json and the lock disagree, which is the difference between
# a reproducible build and a mostly-reproducible one.
COPY package.json package-lock.json ./
RUN npm ci


FROM node:24-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Baked in at build time: NEXT_PUBLIC_* values are inlined into the client
# bundle, so this cannot be supplied at run time. A deployment pointing at a
# different API rebuilds, which is the honest consequence of shipping the URL
# to the browser.
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE

RUN npm run build


FROM node:24-slim AS runtime
WORKDIR /app

RUN useradd --create-home --uid 10001 app

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Standalone output plus the two directories it does not include.
COPY --from=build --chown=app:app /app/.next/standalone ./
COPY --from=build --chown=app:app /app/.next/static ./.next/static
COPY --from=build --chown=app:app /app/public ./public

USER app
EXPOSE 3000

CMD ["node", "server.js"]
