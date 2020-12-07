FROM gcc AS build

COPY . /src

WORKDIR /src

RUN make LDFLAGS="-lm -static"

FROM scratch

COPY --from=build /src/test ./

WORKDIR /data

CMD ["/test"]
