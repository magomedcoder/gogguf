# syntax=docker/dockerfile:1

FROM alpine:3.24.0 AS go-base

ARG GOLANG_VERSION=1.26.0

RUN apk add --no-cache ca-certificates curl git build-base

RUN curl -fsSL "https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xzf -

ENV PATH="/usr/local/go/bin:${PATH}"

WORKDIR /src

COPY go.mod ./
COPY . .

FROM go-base AS cpu-builder

ARG TARGETOS=linux
ARG TARGETARCH=amd64

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w" -o /gogguf ./cmd/gogguf

FROM go-base AS cuda-builder

RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
    go build -tags cuda -trimpath -ldflags="-s -w" -o /gogguf ./cmd/gogguf

FROM alpine:3.24.0 AS runtime-cuda

RUN apk add --no-cache ca-certificates libstdc++

COPY --from=cuda-builder /gogguf /usr/local/bin/gogguf

EXPOSE 8000

VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8000/v1/health >/dev/null 2>&1 || exit 1

ENTRYPOINT ["gogguf"]

CMD ["serve", "-m", "/models/model.gguf", "--addr", "0.0.0.0:8000"]

FROM alpine:3.24.0 AS runtime

RUN apk add --no-cache ca-certificates

COPY --from=cpu-builder /gogguf /usr/local/bin/gogguf

EXPOSE 8000

VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8000/v1/health >/dev/null 2>&1 || exit 1

ENTRYPOINT ["gogguf"]

CMD ["serve", "-m", "/models/model.gguf", "--addr", "0.0.0.0:8000"]
