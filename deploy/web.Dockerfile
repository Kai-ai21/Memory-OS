# syntax=docker/dockerfile:1.7
#
# The UI and the front door (M12.3): Caddy serves the built app, terminates
# HTTPS with a certificate it obtains itself, and proxies /api to the API.
#
# One origin for both, on purpose. The session cookie is SameSite=Lax and the
# live-events stream is an EventSource without credentials; both work
# same-origin and both break the moment the UI and the API are on two sites.

FROM node:22-slim AS build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# Relative: the API is wherever this page was loaded from, under /api.
ENV VITE_API_URL=/api
RUN npm run build

FROM caddy:2.10-alpine AS runtime
COPY deploy/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /web/dist /srv
