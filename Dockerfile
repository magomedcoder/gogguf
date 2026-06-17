
FROM alpine:3.24.0 AS go-base

ARG GOLANG_VERSION=1.26.0

RUN apk update && apk add --no-cache ca-certificates curl git build-base

RUN curl -fsSL "https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xzf -

ENV PATH="/usr/local/go/bin:${PATH}"

WORKDIR /src

COPY go.mod ./
COPY . .

FROM go-base AS cpu-build

RUN set -e; \
    mkdir -p build; \
    for target in \
        "linux amd64" \
        "linux arm64" \
        "windows amd64" \
        "windows arm64" \
        "darwin amd64" \
        "darwin arm64"; \
    do \
        set -- $target; \
        os=$1; arch=$2; \
        ext=""; \
        [ "$os" = "windows" ] && ext=".exe"; \
        out="build/${os}-${arch}/gguf${ext}"; \
        mkdir -p "$(dirname "$out")"; \
        CGO_ENABLED=0 GOOS=$os GOARCH=$arch go build -trimpath -ldflags="-s -w" -o "$out" ./cmd/gguf; \
    done

FROM go-base AS cuda-build

RUN mkdir -p build/linux-amd64

RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -tags cuda -trimpath -ldflags="-s -w" -o build/linux-amd64/gguf-cuda ./cmd/gguf

FROM cpu-build AS release-build

COPY --from=cuda-build /src/build/linux-amd64/gguf-cuda build/linux-amd64/gguf-cuda

FROM release-build AS release

VOLUME ["/out"]

CMD ["sh", "-c", "cp -a build/. /out/"]

FROM cuda-build AS cuda

VOLUME ["/out"]

CMD ["sh", "-c", "cp -a build/. /out/"]

FROM cpu-build

VOLUME ["/out"]

CMD ["sh", "-c", "cp -a build/. /out/"]
